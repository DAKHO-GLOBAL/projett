from setuptools import setup, find_packages

setup(
    name="rl-trading-mt5",
    version="0.1.0",
    description="Reinforcement Learning Trading System with MetaTrader 5",
    author="",
    author_email="",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
        "MetaTrader5>=5.0.33",
        "gymnasium>=0.26.0",
        "torch>=1.9.0",
        "stable-baselines3>=1.6.0",
        "scikit-learn>=1.0.0",
        "ta>=0.10.0",
        "pyfolio>=0.9.2",
        "python-json-logger>=2.0.4",
    ],
    python_requires=">=3.8",
)