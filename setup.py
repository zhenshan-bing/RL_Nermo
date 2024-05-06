from setuptools import setup

setup(name='RL_Nermo',
    version='0.0.1',
    install_requires=[
        'stable-baselines3[extra]==1.4.0',
        'sb3-contrib==1.4.0',
        # 'gym==0.20.0',
        'mujoco-py==2.1.2.14',
        'numpy',
        'torch==1.8.1',
        'optuna==2.10.0',
        'seaborn',
        'plotly',
        'psycopg2',
        'scikit-learn',
        'matplotlib',
        'Cython==0.29.28',
        'tikzplotlib',
    ],
    extras_require={
        "recording": [
            "imageio-ffmpeg",
            "moviepy"
        ]
    },
    python_requires='>=3.8',
    # explicitly declare packages so setuptools does not attempt auto discovery
    packages=[],
)