import test


class MyTestCase(test.TestCase):
    def test_something(self):
        self.assertEqual(True, False)


if __name__ == '__main__':
    test.main()
