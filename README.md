# Regime Detection System

This document describes the new regime detection system implemented in the Actor-Critic model.

## Overview

The system consists of two neural networks that work together to detect market regimes and generate portfolio weights:

1. **IndexRegimeDetector**: Outputs P(regime|x_index) - market index regime probabilities
2. **PortfolioWeightGenerator**: Outputs portfolio weights for each stock under each regime - regime-dependent portfolio construction

## Architecture

### 1. IndexRegimeDetector

A simple MLP-based network that takes market index features and outputs regime probabilities.

```python
class IndexRegimeDetector(nn.Module):
    def __init__(self, input_dim, num_regimes, hidden_dim=64, num_layers=2, dropout=0.1):
        # MLP layers with ReLU activations and dropout
        # Output: softmax probabilities over regimes
```

**Input**: `x_index` - Market index features (batch_size, seq_len, input_dim)
**Output**: `regime_probs` - Regime probabilities (batch_size, num_regimes)

**Features**:
- Simple MLP architecture (no LSTM/HMM complexity)
- Configurable hidden dimensions and layers
- Automatic handling of sequence inputs (takes last timestep)
- Softmax output for proper probability distribution

### 2. PortfolioWeightGenerator

A neural network that generates portfolio weights for each stock under each market regime. Each regime has its own linear head that maps stock features to portfolio weights.

```python
class PortfolioWeightGenerator(nn.Module):
    def __init__(self, stock_feature_dim, index_regime_dim, num_stock_regimes, 
                 hidden_dim=64, num_layers=2, dropout=0.1):
        # Shared feature processing + regime-specific linear heads
        # Output: portfolio weights for each regime
```

**Inputs**:
- `x_stocks`: Individual stock features (batch_size, num_stocks, stock_feature_dim)
- `index_regimes`: Market index regime probabilities (batch_size, index_regime_dim)

**Output**: `portfolio_weights` - Portfolio weights (batch_size, num_stock_regimes, num_stocks)

**Features**:
- **Regime-specific linear heads**: Each regime has its own linear layer that maps features to weights
- **Shared feature processing**: Common MLP processes stock features and regime information
- **Portfolio construction**: Direct output of portfolio weights for each stock under each regime
- **Flexible weight constraints**: Optional tanh activation for bounded weights [-1, 1]
- **Efficient batch processing**: Handles multiple samples and stocks simultaneously

## Integration with Actor

The regime detection system is integrated into the Actor class:

```python
class Actor(nn.Module):
    def __init__(self, ...):
        # Index Regime Detector
        self.index_regime_detector = IndexRegimeDetector(...)
        
        # Stock Regime Detector  
        self.stock_regime_detector = StockRegimeDetector(...)
    
    def forward(self, x):
        # ... process input through LSTM, attention, transformer ...
        
        # Get index regime probabilities
        index_regime_probs = self.index_regime_detector(x_index)
        
        # Get stock regime probabilities
        stock_regime_probs = self.stock_regime_detector(x_stocks, index_regime_probs)
        
        # Get portfolio weights for each regime
        # Concatenate index regime probabilities with stock features
        index_regime_expanded = index_regime_probs.unsqueeze(1).expand(-1, self.N, -1)
        combined_features = torch.cat([x_stocks_reshaped, index_regime_expanded], dim=-1)
        portfolio_weights = self.portfolio_weight_generator(combined_features)
        
        # Combine multiple regime portfolios into a single portfolio
        # Using Hadamard product and regime summation
        stock_regime_probs_t = stock_regime_probs.transpose(1, 2)
        weighted_portfolios = portfolio_weights * stock_regime_probs_t
        final_portfolio = weighted_portfolios.sum(dim=1)
        
        return x_stocks, index_regime_probs, final_portfolio
```

## Usage Examples

### Basic Usage

```python
# Create regime detectors
index_detector = IndexRegimeDetector(
    input_dim=16,
    num_regimes=3,  # bull, bear, sideways
    hidden_dim=64,
    num_layers=2
)

portfolio_generator = PortfolioWeightGenerator(
    stock_feature_dim=16,
    index_regime_dim=3,
    num_stock_regimes=4,  # growth, value, momentum, defensive
    hidden_dim=64,
    num_layers=2
)

# Use with data
x_index = torch.randn(4, 10, 16)  # (batch, seq_len, features)
x_stocks = torch.randn(4, 20, 16)  # (batch, num_stocks, features)

index_probs = index_detector(x_index)
portfolio_weights = portfolio_generator(x_stocks, index_probs)
```

### Using the Complete Actor

```python
# Create Actor with regime detection
actor = Actor(
    seed=42,
    N=20,  # number of stocks
    T=10,  # time steps
    F=4,   # features
    # ... other parameters ...
)

# Forward pass
x = torch.randn(4, 20, 10, 4)  # (batch, stocks, time, features)
x_stocks, index_regimes, final_portfolio = actor(x)
```

## File Structure

- `src/models/actor_critic.py` - Main implementation
- `example_regime_detection.py` - IndexRegimeDetector usage example
- `example_portfolio_weights.py` - PortfolioWeightGenerator usage example
- `example_portfolio_combination.py` - Portfolio combination process demonstration  
- `example_combined_regime_detection.py` - Complete system example
- `test_regime_detector.py` - IndexRegimeDetector tests
- `test_portfolio_weights.py` - PortfolioWeightGenerator tests
- `test_dimension_fix.py` - Dimension fix verification tests

## Key Benefits

1. **Simplicity**: MLP-based approach is easier to train and interpret than complex HMM/LSTM systems
2. **Modularity**: Each component can be used independently or together
3. **Efficiency**: No complex temporal modeling, fast inference
4. **Interpretability**: Clear portfolio weight outputs for regime-dependent strategies
5. **Flexibility**: Configurable architecture for different use cases
6. **Portfolio Construction**: Direct generation of regime-specific portfolio weights

## Training Considerations

- **IndexRegimeDetector**: Use cross-entropy loss with softmax outputs for regime classification
- **PortfolioWeightGenerator**: Use appropriate loss functions for portfolio optimization (e.g., Sharpe ratio, maximum drawdown, etc.)
- Dropout layers help prevent overfitting
- The system can be trained end-to-end or with pre-trained components
- Consider using different learning rates for each component if needed
- Portfolio weights can be trained with financial performance metrics as objectives

## Portfolio Weight Generation

The `PortfolioWeightGenerator` creates regime-specific portfolio weights by:

1. **Shared Feature Processing**: Combines stock features with market regime information through a shared MLP
2. **Regime-Specific Heads**: Each regime has its own linear layer that maps processed features to portfolio weights
3. **Direct Weight Output**: Produces portfolio weights for each stock under each regime

**Output Structure**: `(batch_size, num_regimes, num_stocks)`
- Each `portfolio_weights[batch, regime, :]` gives portfolio weights for all stocks under that regime
- Weights can be used directly for portfolio construction or further normalized

**Weight Constraints**:
- **Bounded**: Use tanh activation for weights in [-1, 1] range
- **Unbounded**: Remove activation for raw linear outputs
- **Custom**: Modify activation functions for specific constraints (e.g., softmax for long-only portfolios)

## Portfolio Combination Process

The system combines multiple regime-specific portfolios into a single portfolio using:

1. **Hadamard Product**: Element-wise multiplication between portfolio weights and stock regime probabilities
2. **Regime Summation**: Sum across regimes to get final portfolio weights

**Mathematical Process**:
```
portfolio_weights: (batch, num_regimes, num_stocks)
stock_regime_probs: (batch, num_stocks, num_regimes)

# Transpose to align dimensions
stock_regime_probs_t = stock_regime_probs.transpose(1, 2)  # (batch, num_regimes, num_stocks)

# Hadamard product
weighted_portfolios = portfolio_weights * stock_regime_probs_t

# Sum across regimes
final_portfolio = weighted_portfolios.sum(dim=1)  # (batch, num_stocks)
```

**Final Output**: `(batch_size, num_stocks)` - Single portfolio weights that incorporate regime information

## Future Enhancements

- Add attention mechanisms for better feature selection
- Implement regime transition modeling
- Add uncertainty quantification
- Support for different regime definitions (sector-based, volatility-based, etc.)
- Portfolio weight regularization and constraints
- Multi-objective optimization for risk-return trade-offs

## Troubleshooting

### Shape Mismatch Issues

If you encounter shape-related errors like:
```
ValueError: not enough values to unpack (expected 3, got 2)
```

This usually indicates a mismatch between the expected input shape for the `StockRegimeDetector` and the actual transformer output shape.

**Root Cause**: The `TransformerEncoder` outputs flattened data `(batch*N, hidden_dim)`, but `StockRegimeDetector` expects `(batch, N, hidden_dim)`.

**Solution**: The Actor automatically reshapes the transformer output before passing it to the stock regime detector:

```python
# Reshape transformer output from (batch*N, hidden_dim) to (batch, N, hidden_dim)
batch_size = x_stocks.shape[0] // self.N
x_stocks_reshaped = x_stocks.reshape(batch_size, self.N, -1)

# Now pass to stock regime detector
stock_regime_probs = self.stock_regime_detector(x_stocks_reshaped, index_regime_probs)
```

**Expected Shapes**:
- Input to Actor: `(batch_size, N, T, F)`
- Output from Actor: 
  - `x_stocks`: `(batch_size * N, hidden_dim)` - Flattened stock features
  - `index_regimes`: `(batch_size, num_regimes)` - Market regime probabilities
  - `stock_regimes`: `(batch_size, N, num_stock_regimes)` - Stock regime probabilities
