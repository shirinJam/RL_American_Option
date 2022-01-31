import QuantLib as ql
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta


class Baseline:
    """This class is reponsible for calculating the option price using baseline models
       Baseline1: The Binomial Tree
       Baseline2: The Black-Scholes model
    """
    def __init__(self, today_date, S0, K, r, sigma, d, T, option_type="put"):

        # initialising the required option parameters
        self.today_date = ql.Date(1, 1, 2021) # Option start-date
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.d = d
        self.T = T
        
        if option_type == "put":
            self.otype = ql.Option.Put
        if option_type == "call":
            self.otype = ql.Option.Call
            
        self.dc = ql.Actual365Fixed()
        self.calendar = ql.NullCalendar()
        self.maturity = ql.Date(31, 12, 2021) # Option end-date

    def baseline_model(self):
        """Calculates the American option price by using Benchmark models

        Returns:
            dict: returns a dictionary with the baseline model name and the corresponding values
        """

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

        binomial_engine = ql.BinomialVanillaEngine(bsm_process, "crr", 100)
        american_option.setPricingEngine(binomial_engine)
        pricing_dict["BinomialTree"] = american_option.NPV()

        return pricing_dict
