"""
Prompts for NL→DSL translation via LLM.

System prompt and few-shot examples for high-quality DSL output.
"""


def build_system_prompt() -> str:
    """
    Build system prompt for DSL generation.

    Returns:
        System prompt string
    """
    return """You are an expert options trading DSL translator. Your job is to convert natural language queries into a precise Domain-Specific Language (DSL) for option pricing, regime analysis, and market outlook.

RULES:
1. Output EXACTLY ONE DSL command line. No explanations, no preamble.
2. Prefer finance-native UPPERCASE verbs: QUOTE, REGIME, RISK, TRANSITION, SCENARIO, EXPLAIN, SURFACE, OUTLOOK
3. Legacy verbs are also accepted: PRICE, COVARIANCE, WHAT_IF
4. Use UPPERCASE for regime names: STABLE, TRANSITION, STRESS, RECOVERY
5. Time format: 30d (days), 4w (weeks), 6m (months), or 0.5 (years as decimal)
6. List format: [item1,item2,item3] with no spaces after commas
7. Capitalization: call/put (lowercase), asset names (lowercase)
8. Prefer finance-native parameter names when possible: spot, strike, expiry, vol, rate, dividend, basket, lookback, target, metrics, backdrop
9. Percent inputs are allowed for vol/rates/dividends, e.g. 20%, 5%, 2%

PREFERRED DSL COMMAND SYNTAX:

QUOTE call/put spot=<spot> strike=<strike> expiry=<time> vol=<vol> [backdrop=<regime>] [rate=<rate>] [dividend=<dividend>]
  Example: QUOTE call spot=100 strike=105 expiry=30d vol=20% backdrop=STRESS rate=5%

REGIME current asset=<asset>
  Example: REGIME current asset=silver

REGIME odds from=<regime> toward=<regime> horizon=<time>
  Example: REGIME odds from=STABLE toward=STRESS horizon=10d

RISK basket=[asset1,asset2,asset3] backdrop=<regime> lookback=<time>
  Example: RISK basket=[silver,gold,dxy] backdrop=STRESS lookback=60d

TRANSITION matrix asset=<asset> normalize=true/false
  Example: TRANSITION matrix asset=silver normalize=true

SCENARIO asset=<asset> target=<regime> [spot=<spot>] [strike=<strike>] [expiry=<time>] metrics=[field1,field2]
  Example: SCENARIO asset=silver target=STRESS metrics=[delta,vega,price]

EXPLAIN regime=<regime> features=[feature1,feature2]
  Example: EXPLAIN regime=TRANSITION features=[ricci_curvature,mst_stress]

SURFACE vol asset=<asset> regime=<regime> strikes=[k1,k2,k3] T=<time>
  Example: SURFACE vol asset=silver regime=STABLE strikes=[0.9,1.0,1.1] T=30d

OUTLOOK asset=<asset> horizon=<time> [backdrop=<regime>]
  Example: OUTLOOK asset=silver horizon=10d backdrop=STRESS

LEGACY DSL COMMAND SYNTAX (still accepted):
PRICE option type=call/put S=<spot> K=<strike> T=<time> sigma=<vol> [regime=<regime>] [r=<rate>] [q=<dividend>]
REGIME prob from=<regime> to=<regime> horizon=<time>
COVARIANCE assets=[asset1,asset2,asset3] regime=<regime> window=<time>
WHAT_IF regime_shift to=<regime> asset=<asset> [S=<spot>] [K=<strike>] [T=<time>] show=[field1,field2]

VOLATILITY SPECIFICATIONS:
- Float or percent: 0.2, 20%, 0.25, etc.
- Regime-adjusted: kan_regime (looks up from KAN network)
- Historical: hist_20d, hist_60d (future feature)

VALID VALUES:
- Assets: silver, gold, dxy
- Regimes: STABLE, TRANSITION, STRESS, RECOVERY, current
- Time suffixes: d (days), w (weeks), m (months)
- Option types: call, put
- Show fields: price, delta, gamma, vega, theta, rho, vol"""


def build_few_shot_examples() -> str:
    """
    Build few-shot examples for in-context learning.

    Returns:
        String with 12 examples covering all DSL verbs
    """
    examples = [
        # Finance-native QUOTE examples
        ("Quote a call with spot 100, strike 105, 30 days to expiry, 20% volatility",
         "QUOTE call spot=100 strike=105 expiry=30d vol=20%"),

        ("Price a silver put in a stress backdrop, 60 days out, 35% vol, 3% risk-free rate",
         "QUOTE put spot=50 strike=48 expiry=60d vol=35% backdrop=STRESS rate=3%"),

        ("Quote an at-the-money silver call using regime-adjusted volatility in recovery, 90 days out",
         "QUOTE call spot=100 strike=100 expiry=90d vol=kan_regime backdrop=RECOVERY"),

        # PRICE examples
        ("Price a call option with spot 100, strike 105, 30 days to expiry, 20% volatility",
         "PRICE option type=call S=100 K=105 T=30d sigma=0.2"),

        ("What's the price of a put with spot 50, strike 48, 60 days out, 35% vol, 3% risk-free rate?",
         "PRICE option type=put S=50 K=48 T=60d sigma=0.35 r=0.03"),

        ("Price a silver call using regime-adjusted volatility in stress mode, spot 100, strike 100, 90 days",
         "PRICE option type=call S=100 K=100 T=90d sigma=kan_regime regime=STRESS"),

        # REGIME examples
        ("What's the current regime for silver?",
         "REGIME current asset=silver"),

        ("What's the probability of transitioning from stable to stress in 10 days?",
         "REGIME prob from=STABLE to=STRESS horizon=10d"),

        ("What are the odds of moving from stable to stress over the next 10 days?",
         "REGIME odds from=STABLE toward=STRESS horizon=10d"),

        # COVARIANCE example
        ("Get the covariance matrix for silver, gold, and DXY in stress regime over 60 days",
         "COVARIANCE assets=[silver,gold,dxy] regime=STRESS window=60d"),

        ("Show the basket risk for silver, gold, and DXY in a stress backdrop over 60 days",
         "RISK basket=[silver,gold,dxy] backdrop=STRESS lookback=60d"),

        # TRANSITION example
        ("Show the transition matrix for silver regimes",
         "TRANSITION matrix asset=silver normalize=true"),

        # WHAT_IF example
        ("Analyze how the Greeks would change if silver shifted to a stress regime. Show delta, vega, and price.",
         "WHAT_IF regime_shift to=STRESS asset=silver show=[delta,vega,price]"),

        ("Run a silver stress scenario and show delta, vega, and price",
         "SCENARIO asset=silver target=STRESS metrics=[delta,vega,price]"),

        # EXPLAIN example
        ("Explain the characteristics of transition regimes using Ricci curvature and MST stress",
         "EXPLAIN regime=TRANSITION features=[ricci_curvature,mst_stress]"),

        # SURFACE example
        ("Generate a volatility surface for silver across different strikes (0.9 to 1.1) in stable regime, 30 days",
         "SURFACE vol asset=silver regime=STABLE strikes=[0.9,0.95,1.0,1.05,1.1] T=30d"),

        ("Give me the near-term outlook for silver over the next 10 days in a stress backdrop",
         "OUTLOOK asset=silver horizon=10d backdrop=STRESS"),

        # Additional examples
        ("Price a call with spot 110, strike 100, 6 months, 25% vol, in recovery regime, 2% dividend",
         "PRICE option type=call S=110 K=100 T=6m sigma=0.25 regime=RECOVERY r=0.05 q=0.02"),

        ("What happens to delta and vega if we move to transition regime? Use spot 100, strike 100, 45 days",
         "WHAT_IF regime_shift to=TRANSITION asset=silver S=100 K=100 T=45d show=[delta,vega]"),
    ]

    formatted = "\n".join([f"Q: {q}\nA: {a}" for q, a in examples])
    return f"EXAMPLES:\n{formatted}\n"


def build_error_recovery_prompt(user_query: str, prev_dsl: str, error_msg: str) -> str:
    """
    Build prompt for error recovery (DSL generation failed validation).

    Args:
        user_query: Original user query
        prev_dsl: Previous DSL output that failed
        error_msg: Error message from validator

    Returns:
        Recovery prompt string
    """
    return f"""Your previous output was invalid. Fix it.

PREVIOUS OUTPUT:
{prev_dsl}

ERROR:
{error_msg}

USER QUERY:
{user_query}

OUTPUT CORRECTED DSL:"""


def build_context_prompt(user_query: str) -> tuple:
    """
    Build full context for DSL generation.

    Args:
        user_query: Natural language query

    Returns:
        (system_prompt, user_message) tuple for chat API
    """
    system = build_system_prompt()
    few_shot = build_few_shot_examples()
    user_message = f"{few_shot}\nQ: {user_query}\nA:"

    return system, user_message
