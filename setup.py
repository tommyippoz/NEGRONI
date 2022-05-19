import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='negroni',
     version='0.2',
     scripts=['negroni-script'] ,
     author="Tommaso Zoppi",
     author_email="tommaso.zoppi@unifi.it",
     description="eNsemblE learninG foR mOdel combiNatIon in python",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/tommyippoz/NEGRONI",
     keywords=['machine learning', 'ensemble learning', 'meta-learning'],
     packages=setuptools.find_packages(),
     classifiers=[
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
     ],
 )
