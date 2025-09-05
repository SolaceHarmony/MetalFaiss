# Contributing to Metal FAISS

Thank you for your interest in contributing to Metal FAISS! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues

Before creating an issue, please check if a similar issue already exists. When reporting issues:

1. **Use a clear and descriptive title**
2. **Describe the exact steps to reproduce the problem**
3. **Provide specific examples and code snippets**
4. **Include your environment details** (Python version, MLX version, OS)
5. **Describe the expected vs actual behavior**

### Suggesting Enhancements

Enhancement suggestions are welcome! Please provide:

1. **Clear description** of the enhancement
2. **Use cases** where it would be helpful
3. **Possible implementation approaches**
4. **Any relevant examples** from other libraries

## 🔧 Development Setup

### Prerequisites

- Python 3.8 or higher
- MLX framework (`pip install mlx`)
- Git

### Setup Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/MetalFaiss.git
cd MetalFaiss

# Install in development mode
cd python
pip install -e .

# Install development dependencies
pip install pytest pytest-cov black isort mypy
```

## 📝 Code Guidelines

### Python Style

- Follow [PEP 8](https://pep8.org/) style guidelines
- Use type hints where appropriate
- Write clear, self-documenting code
- Use meaningful variable and function names

### Code Formatting

We use `black` for code formatting:

```bash
# Format all Python files
black python/metalfaiss/

# Check formatting
black --check python/metalfaiss/
```

### Import Organization

Use `isort` to organize imports:

```bash
# Sort imports
isort python/metalfaiss/

# Check import order
isort --check-only python/metalfaiss/
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m unittest discover metalfaiss.unittest -v

# Run specific test file
python -m unittest metalfaiss.unittest.test_distances -v
```

### Writing Tests

- Add tests for new functionality
- Ensure good test coverage
- Use descriptive test names
- Include edge cases and error conditions

Example test structure:

```python
import unittest
import mlx.core as mx
from metalfaiss import FlatIndex, MetricType

class TestNewFeature(unittest.TestCase):
    
    def setUp(self):
        self.index = FlatIndex(d=128, metric_type=MetricType.L2)
        self.test_data = mx.random.normal((100, 128))
    
    def test_feature_functionality(self):
        # Test the feature
        result = self.index.some_new_method()
        self.assertIsNotNone(result)
        
    def test_feature_edge_cases(self):
        # Test edge cases
        with self.assertRaises(ValueError):
            self.index.some_method_with_invalid_input()
```

## 📚 Documentation

### Docstrings

Use Google-style docstrings:

```python
def search(self, query: mx.array, k: int) -> Tuple[mx.array, mx.array]:
    """Search for k nearest neighbors.
    
    Args:
        query: Query vectors of shape (n_queries, d)
        k: Number of nearest neighbors to return
        
    Returns:
        Tuple of (distances, indices) arrays
        
    Raises:
        ValueError: If k is negative or query has wrong dimensions
    """
```

### README Updates

- Update examples if you change APIs
- Add new features to the feature list
- Keep installation instructions current

## 🚀 Pull Request Process

### Before Submitting

1. **Create a feature branch** from `main`
2. **Make your changes** with clear, focused commits
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Run tests** and ensure they pass
6. **Format code** with `black` and `isort`

### Pull Request Guidelines

1. **Use a clear title** describing the change
2. **Provide detailed description** of what and why
3. **Reference related issues** using keywords (e.g., "Fixes #123")
4. **Include screenshots** for UI changes (if any)
5. **Keep changes focused** - one feature/fix per PR

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added tests for new functionality
- [ ] Updated documentation

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Changes are well documented
```

## 🏗️ Architecture Guidelines

### MLX Integration

- Use MLX arrays for all numerical computations
- Provide NumPy fallbacks where appropriate
- Follow MLX best practices for lazy evaluation
- Use `mx.eval()` appropriately to manage computation graphs

### Code Organization

- Keep modules focused and cohesive
- Use clear separation between index types
- Maintain consistent API patterns across classes
- Follow FAISS naming conventions where applicable

## 📋 Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

## ❓ Questions?

- **General questions**: Use [GitHub Discussions](https://github.com/SolaceHarmony/MetalFaiss/discussions)
- **Bug reports**: Create an [Issue](https://github.com/SolaceHarmony/MetalFaiss/issues)
- **Feature requests**: Create an [Issue](https://github.com/SolaceHarmony/MetalFaiss/issues)

## 🙏 Recognition

Contributors will be:
- Added to the README contributors section
- Acknowledged in release notes for significant contributions
- Invited to join the maintainer team for substantial ongoing contributions

Thank you for contributing to Metal FAISS! 🚀