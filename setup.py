from setuptools import setup

setup(name='dendrotools',
      version='0.1',
      description='Tools for running darknet on tree core images',
      url='http://github.com/cusps/dendrotools',
      author='Jacob Bunzel',
      author_email='bunjake@gmail.com',
      license='NA',
      packages=['dendrotools'],
      zip_safe=False,
      install_requires=[
          "pillow"
      ])
