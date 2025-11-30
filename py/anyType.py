class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

    def __eq__(self, __value: object) -> bool:
        return True

ANY = AnyType("*")