from ast import Num
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PortfolioWeightGenerator(nn.Module):
    """
    Neural network that generates portfolio weights for each stock regime
    Each regime has its own linear head that maps stock features to
    portfolio weights
    """
    def __init__(self, stock_feature_dim, num_stock_regimes,
                 hidden_dim=64, num_layers=2, dropout=0.1):
        super(PortfolioWeightGenerator, self).__init__()

        self.stock_feature_dim = stock_feature_dim
        self.num_stock_regimes = num_stock_regimes
        self.hidden_dim = hidden_dim
        input_dim = stock_feature_dim
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        self.shared_mlp = nn.Sequential(*layers)
        self.regime_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(num_stock_regimes)
        ])
        self.weight_activation = nn.Tanh()

    def forward(self, x):
        """
        Args:
            x_stocks: (batch_size, num_stocks,
                       stock_feature_dim) - Individual stock features
            index_regimes: (batch_size,
                            index_regime_dim) - Market index regime probabilities
        Returns:
            portfolio_weights: (batch_size, num_stock_regimes, num_stocks)
                                    - Portfolio weights for each regime
        """
        batch_size, num_stocks, _ = x.shape
        combined_flat = x.reshape(-1, x.size(-1))
        shared_features = self.shared_mlp(combined_flat)
        regime_weights = []
        for regime_head in self.regime_heads:
            weights = regime_head(shared_features)
            if hasattr(self, 'weight_activation'):
                weights = self.weight_activation(weights)
            weights = weights.reshape(batch_size, num_stocks)
            regime_weights.append(weights)
        portfolio_weights = torch.stack(regime_weights, dim=1)
        return portfolio_weights

class Actor(nn.Module):
    def __init__(self,
                    seed,
                    N, T, F,
                    initial_feature_map_dim=16,
                    hidden_dim=16,
                    num_layers=2,
                    dropout=0.1,
                    transformer_d_model=64,
                    transformer_nhead=8,
                    transformer_layers=2,
                    mlp_hidden_size=256,
                    num_regimes=5
                ):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.num_regimes = num_regimes
        self.feature_mapping = FeatureMapping(F, initial_feature_map_dim)

        self.N, self.T, self.F = N, T, F
        self.initial_feature_map_dim = initial_feature_map_dim

        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=initial_feature_map_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.temporal_attention = TemporalAttention(
            P=hidden_dim,
            alstm_hidden=hidden_dim
        )
        """
        self.index_regime_detector = IndexRegimeDetector(
            input_dim=hidden_dim,
            num_regimes=3,
            hidden_dim=64,
            num_layers=2,
            dropout=dropout
        )
        """
        self.index_regime_detector = NeuralHMMRegimeDetector(
            hidden_dim=hidden_dim,
            num_regimes=num_regimes, mid=192, dropout=0.1
        )
        self.stock_regime_detector = StockRegimeDetector(
            stock_feature_dim=hidden_dim,
            index_regime_dim=num_regimes,
            num_stock_regimes=num_regimes,
            hidden_dim=64,
            num_layers=2,
            dropout=dropout
        )
        self.transformer = TransformerEncoder(
            input_dim=hidden_dim,
            num_stocks=N,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_layers,
            dim_feedforward=mlp_hidden_size,
            dropout=dropout
        )
        self.portfolio_weight_generator = PortfolioWeightGenerator(
            stock_feature_dim=hidden_dim,
            num_stock_regimes=num_regimes,
            hidden_dim=64,
            num_layers=2,
            dropout=dropout
        )

    def forward(self, x, noise=None, verbose=False, prior=None):
        if x.ndimension() == 3:  # Single sample case: (N,T,F)
            x = x.unsqueeze(0)  # Add batch dimension → (batch,N,T,F)
        batch, _, _, _ = x.shape
        reshaped_x = x.reshape(-1, self.F)
        transformed_x = self.feature_mapping(reshaped_x)
        x = transformed_x.reshape(-1, self.T, self.initial_feature_map_dim)

        x, _ = self.lstm(x)
        x = self.temporal_attention(x)
        x = x.reshape(batch, self.N+1, self.hidden_dim)

        x_stocks = x[:,:-1,:]  # Stock features: (batch,N,H)
        x_index = x[:,-1,:]   # Index features: (batch,1,H)

        x_stocks = x_stocks.reshape(-1, x_stocks.size(-1))  # Reshape to (batch×N,H)
        x_index = x_index.reshape(-1, x_index.size(-1))  # Reshape to (batch×1,H)
        x_stocks = self.transformer(x_stocks)
        # Get regime probabilities P(regime|x_index)
        if prior is None:
            prior = torch.full((batch, self.num_regimes), 1.0 / self.num_regimes, device=x.device, dtype=x.dtype)
        index_regime_probs, extras = self.index_regime_detector(x_index, prior)
        # index_regime_probs = self.index_regime_detector(x_index)

        batch_size = x_stocks.shape[0] // self.N
        x_stocks_reshaped = x_stocks.reshape(batch_size, self.N, -1)
        # Get stock regime probabilities P(stock_regimes|index_regimes, x_stock)
        stock_regime_probs = self.stock_regime_detector(x_stocks_reshaped, index_regime_probs)

        # Get portfolio weights for each regime
        x_index_reshaped = x_index.reshape(batch_size, 1, -1)
        x = torch.cat([x_index_reshaped, x_stocks_reshaped], dim=1)
        portfolio_weights = self.portfolio_weight_generator(x)

        stock_regime_probs_t = stock_regime_probs.transpose(1, 2)
        index_regime_probs_expanded = index_regime_probs.unsqueeze(-1)
        regime_probs = torch.cat([index_regime_probs_expanded, stock_regime_probs_t], dim=-1)
        weighted_portfolios = portfolio_weights * regime_probs

        final_portfolio = weighted_portfolios.sum(dim=1)

        if verbose:
            print(index_regime_probs, final_portfolio[0][0])


        if noise is not None:
            noise = torch.from_numpy(noise).float().to(x.device)
            final_portfolio += noise
        final_portfolio = torch.softmax(final_portfolio, dim=1)
        return final_portfolio, index_regime_probs.detach(), prior.detach()
        # return final_portfolio, None, None


class Critic(nn.Module):
    def __init__(self, action_dim,
                    seed,
                    N, T, F,
                    initial_feature_map_dim=16,
                    hidden_dim=16, 
                    num_layers=2, 
                    dropout=0.1,
                    critic_hidden_size=16, critic_use_batch_norm=True,
                    transformer_d_model=64, 
                    transformer_nhead=8, 
                    transformer_layers=2, 
                    mlp_hidden_size=256
                 ):
        super(Critic, self).__init__()
        # Fully connected layers for state and action
        self.fc1 = nn.Linear(N+1, critic_hidden_size) # state
        self.fc_action = nn.Linear(action_dim, critic_hidden_size) # action
        # self.bn = nn.BatchNorm1d(critic_hidden_size) if critic_use_batch_norm else None
        self.bn = nn.LayerNorm(critic_hidden_size) if critic_use_batch_norm else None
        # self.fc_out = nn.Linear(critic_hidden_size, 1) # Q-value
        self.fc_out1 = nn.Linear(critic_hidden_size * 2, critic_hidden_size) # Q-value (doubled due to concatenation)
        self.fc_out2 = nn.Linear(critic_hidden_size, 1) # Q-value (doubled due to concatenation)
        self.use_batch_norm = critic_use_batch_norm

        nn.init.uniform_(self.fc_out2.weight, a=-0.03, b=0.03) 

        self.seed = torch.manual_seed(seed)
        self.feature_mapping = FeatureMapping(F, initial_feature_map_dim)

        self.N, self.T, self.F = N, T, F
        self.initial_feature_map_dim = initial_feature_map_dim

        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(
            input_size=initial_feature_map_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Temporal Attention layer
        self.temporal_attention = TemporalAttention(
            P=hidden_dim,  # Use LSTM hidden size as P
            alstm_hidden=hidden_dim
        )

        # Transformer Encoder
        self.transformer = TransformerEncoder(
            input_dim=hidden_dim,
            num_stocks=N,
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            num_layers=transformer_layers,
            dim_feedforward=mlp_hidden_size,
            dropout=dropout
        )

        self.feature_out = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        if state.ndimension() == 3:  # Single sample case: (21, 10, 4)
            state = state.unsqueeze(0)  # Add batch dimension → (1, 21, 10, 4)
        # feature transformation
        batch, _, _, _ = state.shape
        reshaped_x = state.reshape(-1, self.F)  # Reshape to (batch×N×T, F)
        transformed_x = self.feature_mapping(reshaped_x)
        x = transformed_x.reshape(-1, self.T, self.initial_feature_map_dim)

        x, _ = self.lstm(x)
        x = self.temporal_attention(x)

        x = x.reshape(batch, self.N+1, self.hidden_dim)

        x_stocks = x[:,:-1,:]  # Stock features: (batch*N, hidden_dim)
        x_index = x[:,-1,:]   # Index features: (batch, 1, hidden_dim)

        x_stocks = x_stocks.reshape(-1, x_stocks.size(-1))  # Reshape to (batch×N×T, F)
        x_index = x_index.reshape(-1, x_index.size(-1))  # Reshape to (batch×N×T, F)
        x_stocks = self.transformer(x_stocks)

        x_stocks_reshaped = x_stocks.view(batch, self.N, -1)
        x_index_expanded = x_index.unsqueeze(1)
        x = torch.concat([x_index_expanded, x_stocks_reshaped], axis=1)
        x = self.feature_out(x)
        state = torch.flatten(x, start_dim=1)
        # Fully connected layers for state and action
        t1 = self.fc1(state)
        t2 = self.fc_action(action)

        # x = t1 + t2
        # Concatenate state and action features instead of adding
        x = torch.cat([t1, t2], dim=-1)
        x = self.fc_out1(x)
        if self.use_batch_norm:
            x = self.bn(x)

        x = F.relu(x)
        q_value = self.fc_out2(x)
        return q_value

# TODO
class NeuralHMMRegimeDetector(nn.Module):
    """
    One-step Neural-HMM regime filter (index-level).
    Inputs:
        x: (B, H) feature at time t
        prev_posterior (optional): (B, K) regime probs at t-1.
            If None, uses uniform prior.
    Outputs:
        posterior: (B, K) regime posterior at t
        extras: dict with 'trans': (B,K,K) and 'emit_logits': (B,K)
    """
    def __init__(self, hidden_dim: int, num_regimes: int, mid: int = 128, dropout: float = 0.0):
        super().__init__()
        self.K = num_regimes

        # Transition net: x_t -> logits over KxK (from z_{t-1}=i to z_t=j)
        self.trans_net = nn.Sequential(
            nn.Linear(hidden_dim, mid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mid, num_regimes * num_regimes)
        )

        # Emission net: x_t -> regime-wise emission logits (proxy for log-likelihood)
        # You can replace this with a Gaussian head if you have explicit observations.
        self.emit_net = nn.Sequential(
            nn.Linear(hidden_dim, mid), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(mid, num_regimes)
        )

        # Small bias towards self-transition (optional, helps persistence)
        self.self_trans_bias = nn.Parameter(torch.zeros(num_regimes))

    def forward(self, x: torch.Tensor, prev_posterior: torch.Tensor = None):
        """
        x: (B, H)
        prev_posterior: (B, K) or None
        returns:
            posterior: (B, K)
            extras: dict
        """
        prev_posterior = prev_posterior.reshape(-1, prev_posterior.size(-1))  # Reshape to (batch×1,H)
        B = x.size(0)
        K = self.K

        # ----- Transition matrix P(z_t=j | z_{t-1}=i, x_t)
        trans_logits = self.trans_net(x)                        # (B, K*K)
        trans_logits = trans_logits.view(B, K, K)               # (B, K_prev, K_curr)
        # add self-transition bias on the diagonal
        trans_logits = trans_logits + torch.diag(self.self_trans_bias).unsqueeze(0)  # (B,K,K)
        trans_logprobs = F.log_softmax(trans_logits, dim=-1)    # row-stochastic in log-space

        # ----- Emission "log-likelihood" per regime
        emit_logits = self.emit_net(x)                          # (B, K)
        emit_logprobs = F.log_softmax(emit_logits, dim=-1)      # normalized proxy; replace if you have true llh

        # ----- Prior over z_{t-1}
        prior = prev_posterior.clamp_min(1e-8)              # (B, K)
        prior = prior / prior.sum(dim=-1, keepdim=True)

        log_prior = prior.log()                                  # (B, K)

        # ----- Filtering update in log-space:
        # log pred_j = logsum_i [ prior_i * trans_{i->j} ]
        # = logsumexp_i [ log_prior_i + log_trans_{i->j} ]
        # shapes: (B, K, 1) + (B, K, K) -> reduce over i
        log_pred = torch.logsumexp(log_prior.unsqueeze(-1) + trans_logprobs, dim=1)  # (B, K)

        # posterior ∝ pred * emission  -> add in log-space then normalize
        log_posterior_unnorm = log_pred + emit_logprobs         # (B, K)
        posterior = F.softmax(log_posterior_unnorm, dim=-1)     # (B, K)

        extras = {
            "trans": trans_logprobs.exp(),      # (B,K,K)
            "emit_logits": emit_logits          # (B,K)
        }
        return posterior, extras

class IndexRegimeDetector(nn.Module):
    """
    Simple MLP-based regime detector that outputs P(regime|x_index)
    """
    def __init__(self, input_dim, num_regimes, hidden_dim=64, num_layers=2, dropout=0.1):
        super(IndexRegimeDetector, self).__init__()
        self.input_dim = input_dim
        self.num_regimes = num_regimes
        self.hidden_dim = hidden_dim
        # Build MLP layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, num_regimes))
        self.mlp = nn.Sequential(*layers)
        # Softmax for probability output
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x_index):
        """
        Args:
            x_index: (batch,H) - Market index features
        Returns:
            regime_probs: (batch,R) - P(regime|x_index)
        """
        # If input has sequence dimension, take the last timestep
        if x_index.dim() == 3:
            x_index = x_index[:, -1, :]  # (batch_size, input_dim)
        # Pass through MLP
        logits = self.mlp(x_index)
        # Apply softmax to get probabilities
        regime_probs = self.softmax(logits)
        return regime_probs

class StockRegimeDetector(nn.Module):
    """
    Neural network that outputs P(stock_regimes|index_regimes, x_stock)
    Takes both market index regime probabilities and individual stock features as input
    """
    def __init__(self, stock_feature_dim, index_regime_dim, num_stock_regimes, 
                 hidden_dim=64, num_layers=2, dropout=0.1):
        super(StockRegimeDetector, self).__init__()
        self.stock_feature_dim = stock_feature_dim
        self.index_regime_dim = index_regime_dim
        self.num_stock_regimes = num_stock_regimes
        self.hidden_dim = hidden_dim
        # Input dimension: stock features + index regime probabilities
        input_dim = stock_feature_dim + index_regime_dim
        # Build MLP layers
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, num_stock_regimes))
        self.mlp = nn.Sequential(*layers)
        # Softmax for probability output
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_stocks, index_regimes):
        """
        Args:
            x_stocks: (batch_size, num_stocks, stock_feature_dim) - Individual stock features
            index_regimes: (batch_size, index_regime_dim) - Market index regime probabilities
        Returns:
            stock_regime_probs: (batch_size, num_stocks, num_stock_regimes) - P(stock_regime|index_regime, x_stock)
        """
        batch_size, num_stocks, _ = x_stocks.shape
        # Expand index regimes to match stock dimensions
        # index_regimes: (batch_size, 1, index_regime_dim) -> (batch_size, num_stocks, index_regimes_dim)
        index_regimes_expanded = index_regimes.unsqueeze(1).expand(-1, num_stocks, -1)
        # Concatenate stock features with index regime probabilities
        # combined: (batch_size, num_stocks, stock_feature_dim + index_regime_dim)
        combined_features = torch.cat([x_stocks, index_regimes_expanded], dim=-1)
        # Reshape for batch processing through MLP
        # combined_flat: (batch_size * num_stocks, stock_feature_dim + index_regime_dim)
        combined_flat = combined_features.reshape(-1, combined_features.size(-1))
        # Pass through MLP
        logits = self.mlp(combined_flat)
        # Apply softmax to get probabilities
        regime_probs = self.softmax(logits)
        # Reshape back to (batch_size, num_stocks, num_stock_regimes)
        stock_regime_probs = regime_probs.reshape(batch_size, num_stocks, self.num_stock_regimes)
        return stock_regime_probs

class FeatureMapping(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(self.__class__, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, inputs):
        return torch.tanh(self.linear(inputs))


class TemporalAttention(nn.Module):
    def __init__(self, P, alstm_hidden=32):
        super(self.__class__, self).__init__()
        self.P = P
        self.alstm_hidden = alstm_hidden
        
        # Query, Key, Value projections
        self.query_proj = nn.Linear(self.P, self.P)
        self.key_proj = nn.Linear(self.P, self.P)
        self.value_proj = nn.Linear(self.P, self.P)
        
        # Scaling factor for attention
        self.scale = self.P ** 0.5

    def forward(self, encoded_inputs):
        batch_size, seq_len, hidden_dim = encoded_inputs.shape
        
        # Use the last hidden state as query
        query = self.query_proj(encoded_inputs[:, -1:, :])  # (batch_size, 1, P)
        
        # Use all hidden states except the last as keys and values
        keys = self.key_proj(encoded_inputs[:, :-1, :])     # (batch_size, seq_len-1, P)
        values = self.value_proj(encoded_inputs[:, :-1, :]) # (batch_size, seq_len-1, P)
        
        # Compute attention scores
        # query: (batch_size, 1, P), keys: (batch_size, seq_len-1, P)
        # scores: (batch_size, 1, seq_len-1)
        scores = torch.matmul(query, keys.transpose(-2, -1)) / self.scale
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)  # (batch_size, 1, seq_len-1)
        
        # Apply attention weights to values
        # attention_weights: (batch_size, 1, seq_len-1), values: (batch_size, seq_len-1, P)
        # attended_values: (batch_size, 1, P)
        attended_values = torch.matmul(attention_weights, values)
        
        # Squeeze the middle dimension and return the weighted sum
        attended_values = attended_values.squeeze(1)  # (batch_size, P)
        return attended_values

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(self.__class__, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model) for batch_first=True
        # Add positional encoding to the sequence dimension
        x = x + self.pe[:x.size(1), :].unsqueeze(0)  # Add to seq_len dimension
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_stocks, d_model=64, nhead=8, num_layers=2, dim_feedforward=256, dropout=0.1):
        super(self.__class__, self).__init__()
        self.input_dim = input_dim
        self.num_stocks = num_stocks
        self.d_model = d_model
        
        # Input projection to match d_model
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding (since we're treating the input as a sequence)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection back to desired dimension
        self.output_projection = nn.Linear(d_model, input_dim)
        
    def forward(self, x):
        total_samples, F = x.shape
        N = self.num_stocks
        batch = total_samples // N
        
        # Reshape to (batch, N, F)
        x = x.reshape(batch, N, F)
        
        x = self.input_projection(x)  # (batch, N, d_model)
        x = self.pos_encoder(x)  # (batch, N, d_model)
        x = self.transformer_encoder(x)  # (batch, N, d_model)
        x = self.output_projection(x)  # (batch, N, F)
        x = x.reshape(total_samples, F)
        
        return x
