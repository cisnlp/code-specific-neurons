import setuptools

with open('requirements.txt') as f:
    required = f.read().splitlines()

with open("README.md", encoding='utf-8-sig') as fh:
    long_description = fh.read()

setuptools.setup(
    name='code-lens',
    version="1.0",
    description="Tools for understanding how Code LLMs predictions are built layer-by-layer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/cisnlp/code-lens',
    packages=setuptools.find_packages(),
    package_data={
        'code-lens.utils': ['keywords/*.json']
    },
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License'
    ],
    install_requires=required,
    author='Amir Hossein Kargaran',
    author_email='kargaranamir@gmail.com'
)