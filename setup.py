from setuptools import setup, find_packages
import glob

def get_bash_scripts():
    return glob.glob("bin/*.sh")

setup(
    name="facebook_muse",
    packages=find_packages(),
    python_requires=">=2.7",
    install_requires=["faiss",
                      "numpy",
                      "scipy",
                      "torch"],
    extras_require={
        "dev" : ["jupyter"]
    },
    scripts=get_bash_scripts(),
    entry_points={
        'console_scripts' : [
            "supervised=facebook_muse.supervised:main",
            "unsupervised=facebook_muse.unsupervised:main",
            "evaluation=facebook_muse.evaluation:main"
        ]
    }
)
