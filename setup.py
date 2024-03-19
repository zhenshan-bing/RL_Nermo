from setuptools import setup

setup(name='ba_nermo_rl_locomotion',
    version='0.0.1',
    install_requires=[
        'stable-baselines3[extra]==1.4.0',
        'sb3-contrib==1.4.0',
        # 'gym>=0.20.0,<1',
        'mujoco-py>=2.0,<2.1',
        'numpy',
        'torch>=1.8.1',
        'optuna',
        'seaborn',
        'plotly',
        'psycopg2',
        'scikit-learn',
        'matplotlib'
    ],
    extras_require={
        "recording": [
            "imageio-ffmpeg",
            "moviepy"
        ]
    },
    python_requires='>=3.6',
    # explicitly declare packages so setuptools does not attempt auto discovery
    packages=[],
)