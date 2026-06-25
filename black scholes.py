
#===============================================================================
# BLACK_SCHOLES_OPTION_PRICER
#===============================================================================


#-------------------------------------------------------------------------------
#imports
#-------------------------------------------------------------------------------


import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------
#BLACK_SCHOLES_PRICING_FUNCTION
#-------------------------------------------------------------------------------

#Black Scholes Model
def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
Black-Scholes option pricing formula.

parameters:
    S : current stock price (spot)
    K : strike price
    T : time to expiry in years (e.g. 0.5 = 6 months)
    r : risk-free interest rate (e.g. 0.05 = 5%) 
    sigma : volatility (e.g. 0.2 = 20%)
    option_type : 'call' or 'put'

    returns:
        price of the option
    """

#d1 and d2 are intermediate values the formula needs \
    d1 = (np.log (S/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = S * norm.cdf(d1) - K * np.exp(-r*T)*norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")
    return price

#TEST
price =black_scholes(S=100, K=100,T=1,r=0.05,sigma=0.2, option_type='call')
print(f"Call price: ${price:.4f}")

#ANNOTATIONS
# The stock is currently priced at $100
# have the right to buy it at $100 (the strike)
# In 1 year's time
# Risk-free rate is 5%
# The stock moves around by 20% per year (volatility)

    # S = current stock price (spot price)
    # e.g. S=100 means the stock is trading at $100 right now

    # K = strike price
    # the price you have agreed in advance to buy (call) or sell (put) the stock at
    # e.g. K=110 means you have the right to buy at $110 no matter what the stock does

    # T = time to expiry, measured in years
    # e.g. T=1 is one year, T=0.5 is 6 months, T=0.25 is 3 months
    # the longer the time, the more expensive the option (more time for stock to move)

    # r = risk-free interest rate
    # the return you could get by doing nothing and putting money in a safe investment
    # e.g. r=0.05 means 5% — roughly the US government bond rate
    # used to discount the future payoff back to today's value

    # sigma = volatility of the stock, measured as a decimal
    # how much the stock price moves around per year
    # e.g. sigma=0.2 means 20% — a typical stock
    # higher sigma = more uncertainty = more expensive option

    # option_type = whether this is a call or a put
    # call = right to BUY the stock at the strike price
    # put  = right to SELL the stock at the strike price


#-------------------------------------------------------------------------------
# GREEKS
#-------------------------------------------------------------------------------
def greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculates the five main Greeks.
    Returns a dictionary with delta, gamma, theta, vega, rho.
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100

    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = ((-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                  - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365)
        rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = ((-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                  + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365)
        rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega' : vega,
        'rho'  : rho
    }

#TEST
g=greeks(S=100, K=100,T=1,r=0.05,sigma=0.2, option_type='call')
for name,value in g.items():
    print(f"{name:>6}:{value:.6}")

#ANNOTATIONS
def greeks(S, K, T, r, sigma, option_type='call'):

    # S = current stock price
    # K = strike price
    # T = time to expiry in years
    # r = risk-free interest rate
    # sigma = volatility of the stock
    # option_type = 'call' or 'put'

    # d1 and d2 are the same intermediate values from Black-Scholes
    # the Greeks are just derivatives of the same formula
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # gamma and vega are identical for calls and puts
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))  # delta change per $1 spot move
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100      # price change per 1% vol move

    if option_type == 'call':
        delta = norm.cdf(d1)                          # between 0 and 1 for calls
        theta = ((-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                  - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365)  # daily decay
        rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100        # per 1% rate move
    else:
        delta = norm.cdf(d1) - 1                      # between -1 and 0 for puts
        theta = ((-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                  + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365)
        rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100      # negative for puts

    return {
        'delta': delta,  # price change per $1 spot move
        'gamma': gamma,  # delta change per $1 spot move
        'theta': theta,  # price lost per day
        'vega' : vega,   # price change per 1% vol move
        'rho'  : rho     # price change per 1% rate move
    }

#-------------------------------------------------------------------------------
# BINOMIAL TREE
#-------------------------------------------------------------------------------

def binomial_tree(S,K,T,r,sigma,option_type='call', N=100):
    """
    Prces a European option using a binomial tree with N steps
    converges to Black-Scholes as N increases.
    """

    dt= T / N
    u = np.exp(sigma*np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d)/(u - d)

    # Build terminal stock prices at expiry
    # After N steps, price = S * u^(N-i)*d^i for i=0,1,...,N
    i = np.arange(N+1)
    terminal_prices = S * (u**(N-i) * (d**i))

    #calculate payoffs at expiry
    if option_type =='call':
        payoffs = np.maximum(terminal_prices - K,0)
    else:
        payoffs = np.maximum(K-terminal_prices,0)

    #work backwards through the tree, discounting at each step
    discount = np.exp(-r*dt)
    for _ in range (N):
        payoffs = discount * (p*payoffs[:-1]+(1-p)* payoffs[1:])

    return payoffs[0]

#TEST
binom_price=binomial_tree(S=100, K=100,T=1,r=0.05,sigma=0.2, option_type='call', N=100)
print(f"Binomial price(N=100): ${binom_price:.4f}")

    
def print_summary(S, K, T, r, sigma, option_type='call'):

    # run all three models
    price = black_scholes(S, K, T, r, sigma, option_type)
    g     = greeks(S, K, T, r, sigma, option_type)
    binom = binomial_tree(S, K, T, r, sigma, option_type, N=200)

    # work out if the option is in or out of the money
    if option_type == 'call':
        moneyness = 'in the money' if S > K else 'out of the money'
    else:
        moneyness = 'in the money' if S < K else 'out of the money'

    print("=" * 40)
    print(f"  {option_type.upper()} OPTION SUMMARY")
    print("=" * 40)
    print(f"  Spot:     ${S}")
    print(f"  Strike:   ${K}  ({moneyness})")
    print(f"  Expiry:   {T} year(s)")
    print(f"  Rate:     {r*100:.1f}%")
    print(f"  Vol:      {sigma*100:.1f}%")
    print("-" * 40)
    print(f"  Black-Scholes price : ${price:.4f}")
    print(f"  Binomial (N=200)    : ${binom:.4f}")
    print(f"  Difference          : ${abs(price-binom):.6f}")
    print("-" * 40)
    print("  GREEKS")
    print(f"  Delta : {g['delta']:>10.6f}  (per $1 spot move)")
    print(f"  Gamma : {g['gamma']:>10.6f}  (per $1 spot move)")
    print(f"  Theta : {g['theta']:>10.6f}  (per day)")
    print(f"  Vega  : {g['vega']:>10.6f}  (per 1% vol move)")
    print(f"  Rho   : {g['rho']:>10.6f}  (per 1% rate move)")
    print("=" * 40)


def plot_all(S, K, T, r, sigma, option_type='call'):

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(f'Options Analysis — {option_type.capitalize()}  '
                 f'S={S}, K={K}, T={T}yr, r={r}, σ={sigma}', fontsize=13)

#-------------------------------------------------------------------------------
#Chart 1: Price vs Spot 
#-------------------------------------------------------------------------------
   
    spot_range = np.linspace(S * 0.5, S * 1.5, 200)
    prices     = [black_scholes(s, K, T, r, sigma, option_type) for s in spot_range]
    intrinsic  = [max(0, s - K) if option_type == 'call' else max(0, K - s) for s in spot_range]

    ax = axes[0, 0]
    ax.plot(spot_range, prices,    label='B-S price',       color='#534AB7', linewidth=2)
    ax.plot(spot_range, intrinsic, label='Intrinsic value', color='#888', linestyle='--', linewidth=1)
    ax.axvline(S, color='#D85A30', linestyle=':', linewidth=1, label=f'Current spot ${S}')
    ax.set_title('Price vs spot', fontsize=11)
    ax.set_xlabel('Spot price ($)')
    ax.set_ylabel('Option price ($)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
#-------------------------------------------------------------------------------
# Chart 2: Price vs Volatility
#-------------------------------------------------------------------------------
    vol_range = np.linspace(0.05, 0.8, 200)
    prices_v  = [black_scholes(S, K, T, r, v, option_type) for v in vol_range]

    ax = axes[0, 1]
    ax.plot(vol_range * 100, prices_v, color='#0F6E56', linewidth=2)
    ax.axvline(sigma * 100, color='#D85A30', linestyle=':', linewidth=1, label=f'Current vol {sigma*100:.0f}%')
    ax.set_title('Price vs volatility', fontsize=11)
    ax.set_xlabel('Volatility (%)')
    ax.set_ylabel('Option price ($)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

#-------------------------------------------------------------------------------
# Chart 3: Delta and Gamma vs Spot 
#-------------------------------------------------------------------------------
    
    deltas = [greeks(s, K, T, r, sigma, option_type)['delta'] for s in spot_range]
    gammas = [greeks(s, K, T, r, sigma, option_type)['gamma'] for s in spot_range]

    ax = axes[1, 0]
    ax2 = ax.twinx()
    ax.plot(spot_range,  deltas, color='#534AB7', linewidth=2, label='Delta')
    ax2.plot(spot_range, gammas, color='#D85A30', linewidth=2, label='Gamma', linestyle='--')
    ax.axvline(S, color='gray', linestyle=':', linewidth=1)
    ax.set_title('Delta & Gamma vs spot', fontsize=11)
    ax.set_xlabel('Spot price ($)')
    ax.set_ylabel('Delta', color='#534AB7')
    ax2.set_ylabel('Gamma', color='#D85A30')
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)

#-------------------------------------------------------------------------------
#  Chart 4: Binomial convergence 
#-------------------------------------------------------------------------------
    
    step_sizes   = list(range(5, 205, 5))
    binom_prices = [binomial_tree(S, K, T, r, sigma, option_type, N=n) for n in step_sizes]
    bs_price     = black_scholes(S, K, T, r, sigma, option_type)

    ax = axes[1, 1]
    ax.plot(step_sizes, binom_prices, color='#0F6E56', linewidth=2, label='Binomial tree')
    ax.axhline(bs_price, color='#534AB7', linestyle='--', linewidth=1.5, label=f'B-S = ${bs_price:.3f}')
    ax.set_title('Binomial convergence to B-S', fontsize=11)
    ax.set_xlabel('Number of steps (N)')
    ax.set_ylabel('Option price ($)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('options_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Chart saved as options_analysis.png")

    print_summary(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
plot_all(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
