class Likert:

    def __init__(self, n: int) -> None:
        self._n = n
        self._range = self._n - 1
        self._array = list(range(1, self._n + 1))

    @property
    def width(self) -> float:
        return round((self._range / self._n), 2)

    @property
    def interval(self) -> list[list[float, float]]:
        arr1 = []

        for i in self._array:
            arr1.append(round(i + self.width, 2))

            for j in arr1:
                arr1.append(round(j + self.width, 2))

                if len(arr1) == (len(self._array) - 1):
                    break

            break

        arr1.insert(0, 1)
        arr2 = []

        for i in arr1[1:]:
            arr2.append(round(i - 0.01, 2))

        arr2.append(max(self._array))

        return list(map(list, zip(arr1, arr2)))


if __name__ == "__main__":

    likert = Likert(5)
    print(likert.interval)
