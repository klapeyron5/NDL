from numbers import Integral


class UnitTests:

    def test(self, f, input, truth):
        self.assert_input(input)
        self.assert_output(truth)
        output = f(input)
        self.assert_output(output)
        self.compare_output(truth, output)

    def assert_input(self, input):
        assert all([isinstance(x, Integral) for x in input])

    def assert_output(self, output):
        self.assert_input(output)

    def compare_output(self, truth, output):
        assert len(truth) == len(output)
        assert all([x == y for x, y in zip(truth, output)])

    def all_test(self, f):
        self.test1(f)
        self.test2(f)
        self.test3(f)
        self.test4(f)
        self.test5(f)
        self.test6(f)
        self.test7(f)
        self.test8(f)
        self.test9(f)
        self.test10(f)
        self.test11(f)
        self.test12(f)
        self.test13(f)

    def test1(self, f):
        A = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
        truth = [4, -1, 2, 1]
        self.test(f, A, truth)

    def test2(self, f):
        A = [-2, -3, -1, -5]
        truth = [-1]
        self.test(f, A, truth)

    def test3(self, f):
        A = [-2, -3, -1, -5, 0]
        truth = [0]
        self.test(f, A, truth)

    def test4(self, f):
        A = [-1, 2, 3, -9]
        truth = [2, 3]
        self.test(f, A, truth)

    def test5(self, f):
        A = [-1, 2, 3, -9, 11]
        truth = [11]
        self.test(f, A, truth)

    def test6(self, f):
        A = [2, -8, 5, -1, 2, -3, 2]
        truth = [5, -1, 2]
        self.test(f, A, truth)

    def test7(self, f):
        A = [-22, -33, -11, -5, -100]
        truth = [-5]
        self.test(f, A, truth)

    def test8(self, f):
        A = [-100]
        truth = [-100]
        self.test(f, A, truth)

    def test9(self, f):
        A = [-100, -200]
        truth = [-100]
        self.test(f, A, truth)

    def test10(self, f):
        A = [-100, 0]
        truth = [0]
        self.test(f, A, truth)

    def test11(self, f):
        A = [11, 2, -1]
        truth = [11, 2]
        self.test(f, A, truth)

    def test12(self, f):
        A = [11, -2]
        truth = [11]
        self.test(f, A, truth)

    def test13(self, f):
        A = [11, 2]
        truth = [11, 2]
        self.test(f, A, truth)


from task1.main import findMaxSubArray
ut = UnitTests()
ut.all_test(findMaxSubArray)
