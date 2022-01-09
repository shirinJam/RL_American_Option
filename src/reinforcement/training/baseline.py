import QuantLib as ql
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


class Baseline:
    def __init__(self, today_date, S0, K, r, sigma, d, T):
        """[summary]

        Args:
            today_date ([type]): [description]
            S0 ([type]): [description]
            K ([type]): [description]
            r ([type]): [description]
            sigma ([type]): [description]
            d ([type]): [description]
            T ([type]): [description]
        """

        self.today_date = ql.Date(1, 1, 2019)
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.d = d
        self.T = T

        self.otype = ql.Option.Put
        self.dc = ql.Actual365Fixed()
        self.calendar = ql.NullCalendar()
        self.maturity = ql.Date(
            31, 12, 2019
        )  # self.today_date + relativedelta(years=1)

    def baseline_model(self):

        ql.Settings.instance().evaluationDate = self.today_date
        payoff = ql.PlainVanillaPayoff(self.otype, self.K)

        european_exercise = ql.EuropeanExercise(self.maturity)
        european_option = ql.VanillaOption(payoff, european_exercise)

        american_exercise = ql.AmericanExercise(self.today_date, self.maturity)
        american_option = ql.VanillaOption(payoff, american_exercise)

        d_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(self.today_date, self.d, self.dc)
        )
        r_ts = ql.YieldTermStructureHandle(
            ql.FlatForward(self.today_date, self.r, self.dc)
        )
        sigma_ts = ql.BlackVolTermStructureHandle(
            ql.BlackConstantVol(self.today_date, self.calendar, self.sigma, self.dc)
        )
        bsm_process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(ql.SimpleQuote(self.S0)), d_ts, r_ts, sigma_ts
        )

        pricing_dict = {}

        bsm73 = ql.AnalyticEuropeanEngine(bsm_process)
        european_option.setPricingEngine(bsm73)
        pricing_dict["BlackScholesEuropean"] = european_option.NPV()

        analytical_engine = ql.BaroneAdesiWhaleyApproximationEngine(bsm_process)
        american_option.setPricingEngine(analytical_engine)
        pricing_dict["BawApproximation"] = american_option.NPV()

        binomial_engine = ql.BinomialVanillaEngine(bsm_process, "crr", 100)
        american_option.setPricingEngine(binomial_engine)
        pricing_dict["BinomialTree"] = american_option.NPV()

        return pricing_dict
