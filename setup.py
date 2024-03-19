import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()



__version__ = "0.0.0"

REPO_NAME = "JdVVNL"
AUTHOR_USER_NAME = "bhattshubham1401"
SRC_REPO = "JdVVNL"
AUTHOR_EMAIL = "shubhambhatt687@gmail.com"


setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for ml app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)