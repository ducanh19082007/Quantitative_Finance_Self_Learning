import math


class ZeroCouponBond:

    def __init__(self, face_value, maturity, interest_rate):
        self.face_value = face_value
        self.maturity = maturity
        self.r = interest_rate / 100  # market rate (discrete)

    def price(self):
        return self.face_value / (1 + self.r) ** self.maturity

    def price_continuous(self):
        return self.face_value * math.exp(-self.r * self.maturity)

    def ytm(self, market_price):
        return (self.face_value / market_price) ** (1 / self.maturity) - 1

    def ytm_continuous(self, market_price):
        return -math.log(market_price / self.face_value) / self.maturity


class CouponBond:

    def __init__(self, face_value, coupon_rate, maturity):
        self.face_value = face_value
        self.coupon_rate = coupon_rate / 100
        self.maturity = maturity

    def coupon(self):
        return self.face_value * self.coupon_rate

    def price(self, ytm):
        c = self.coupon()
        F = self.face_value
        n = self.maturity

        pv = 0.0

        for t in range(1, n + 1):
            cf = c if t < n else c + F
            pv += cf / (1 + ytm) ** t

        return pv

    def price_continuous(self, r):
        c = self.coupon()
        F = self.face_value
        n = self.maturity

        return (c / r) * (1 - math.exp(-r * n)) + F * math.exp(-r * n)

    def ytm(self, market_price, eps=1e-8):
        low, high = 0.0, 1.0

        def price(y):
            return self.price(y)

        for _ in range(100):
            mid = (low + high) / 2
            p = price(mid)

            if abs(p - market_price) < eps:
                return mid

            if p > market_price:
                low = mid
            else:
                high = mid

        return (low + high) / 2

    def ytm_continuous(self, market_price, eps=1e-8):
        low, high = 1e-6, 1.0

        def price(r):
            return self.price_continuous(r)

        for _ in range(100):
            mid = (low + high) / 2
            p = price(mid)

            if abs(p - market_price) < eps:
                return mid

            if p > market_price:
                low = mid
            else:
                high = mid

        return (low + high) / 2

    def macaulay_duration(self, ytm):
        c = self.coupon()
        F = self.face_value
        n = self.maturity

        price = self.price(ytm)
        weighted = 0.0

        for t in range(1, n + 1):
            cf = c if t < n else c + F
            pv = cf / (1 + ytm) ** t
            weighted += t * pv

        return weighted / price

    def modified_duration(self, ytm):
        return self.macaulay_duration(ytm) / (1 + ytm)


if __name__ == "__main__":

    print("===== ZERO COUPON TEST =====")

    zcb = ZeroCouponBond(1000, 2, 5)

    p = zcb.price()
    pc = zcb.price_continuous()

    print("Discrete price:", round(p, 6))
    print("Continuous price:", round(pc, 6))

    print("Recovered YTM:", zcb.ytm(p) * 100, "%")
    print("Recovered cont YTM:", zcb.ytm_continuous(pc) * 100, "%")


    print("\n===== COUPON BOND TEST =====")
    
    bond = CouponBond(1000, 10, 3)

    market_y = 0.06
    market_price = bond.price(market_y)

    print("Market price:", round(market_price, 6))

    ytm = bond.ytm(market_price)
    print("Recovered YTM:", ytm * 100, "%")


    print("\n===== DURATION TEST =====")

    Dm = bond.macaulay_duration(ytm)
    Dmod = bond.modified_duration(ytm)

    print("Macaulay Duration:", (Dm), "out of 3 days" )
    print("Modified Duration:", Dmod, "out of 3 days")