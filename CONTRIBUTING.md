# Contributing to Multi-Task NLP API

First off, thanks for taking the time to contribute! 🎉

We welcome contributions from the community to help make this project better. Whether it's fixing bugs, improving documentation, or proposing new features, your help is appreciated.

## 🛠️ How to Contribute

### 1. Verification
Before starting, ensure you have the project setup requirements:
- Docker Engine & Docker Compose
- Python 3.10+ (if running locally without Docker)

### 2. Fork & Clone
1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/multitask-nlp-api.git
   cd multitask-nlp-api
   ```

### 3. Create a Branch
Create a descriptive branch for your changes:
```bash
git checkout -b feature/amazing-new-feature
# or
git checkout -b fix/critical-bug-fix
```

### 4. Make Your Changes
- **Code Style**: We follow PEP 8. Please use `flake8` or `black` to format your code.
- **Type Hints**: Ensure all functions have proper type hints (Python `typing`).
- **Commits**: Write clear, concise commit messages.

### 5. Tests
If you add code, please add tests. We use `pytest`.
```bash
# Run tests inside the container
docker-compose run results-api pytest
```

### 6. Pull Request
1. Push your branch to GitHub.
2. Submit a Pull Request (PR) to the `main` branch.
3. Describe what your PR does and link to any relevant issues.

---

## 🐛 Reporting Bugs

If you find a bug, please open an issue and include:
- Steps to reproduce
- Expected behavior vs. actual behavior
- Logs or screenshots (if applicable)

## 📄 License
By contributing, you agree that your contributions will be licensed under the MIT License.
