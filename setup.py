import setuptools

setuptools.setup(
        name = 'ante',
        version = '0.1.0',
        packages = setuptools.find_packages(),
        entry_points = {
            'console_scripts':[
                'tunimi = tunimi.main:main',
                'ante-train = ante.train:main',
                'ante-generate = ante.generate:main',
                ]},)
