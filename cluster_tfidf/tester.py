class PackageTester:
    def __init__(self):
        self.output = 'Hello World!'

    def result(self):
        print(self.output)

if __name__=='__main__':
    test = PackageTester()
    test.result()