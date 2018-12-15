from setuptools import setup, find_packages


if __name__ == "__main__":
    setup(
        name="geometry-score",
        author="Valentin Khrulkov",
        packages=find_packages(),
        install_requires=['numpy', 'scipy', 'matplotlib', 'Cython', 'six']
    )
