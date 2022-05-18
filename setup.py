import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='negroni',
     version='0.1',
     scripts=['negroni'] ,
     author="Tommaso Zoppi",
     author_email="tommaso.zoppi@unifi.it",
     description="eNsemblE learninG foR mOdel combiNatIon in python",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/tommyippoz/NEGRONI",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
