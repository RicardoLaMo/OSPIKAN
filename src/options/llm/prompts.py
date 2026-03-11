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
    return """You are an expert options trading DSL translator. Your job is to convert natural language queries into a precise Domain-Specific Language (DSL) for option pricing and regime analysis.

RULES:
1. Output EXACTLY ONE DSL command line. No explanations, no preamble.
2. Use UPPERCASE for verbs: PRICE, REGIME, COVARIANCE, TRANSITION, WHAT_IF, EXPLAIN, SURFACE
3. Use UPPERCASE for regime names: STABLE, TRANSITION, STRESS, RECOVERY
4. Time format: 30d (days), 4w (weeks), 6m (months), or 0.5 (years as decimal)
5. List format: [item1,item2,item3] with no spaces after commas
6. Capitalization: call/put (lowercase), asset names (lowercase)
7. Always use parameter names exactly as shown in examples: type=, S=, K=, T=, sigma=, regime=, r=, q=, asset=, etc.

DSL COMMAND SYNTAX:

PRICE option type=call/put S=<spot> K=<strike> T=<time> sigma=<vol> [regime=<regime>] [r=<rate>] [q=<dividend>]
  Example: PRICE option type=call S=100 K=105 T=30d sigma=0.2 regime=STRESS r=0.05

REGIME current asset=<asset>
  Example: REGIME current asset=silver

REGIME prob from=<regime> to=<regime> horizon=<time>
  Example: REGIME prob from=STABLE to=STRESS horizon=10d

COVARIANCE assets=[asset1,asset2,asset3] regime=<regime> window=<time>
  Example: COVARIANCE assets=[silver,gold,dxy] regime=STRESS window=60d

TRANSITION matrix asset=<asset> normalize=true/false
  Example: TRANSITION matrix asset=silver normalize=true

WHAT_IF regime_shift to=<regime> asset=<asset> [S=<spot>] [K=<strike>] [T=<time>] show=[field1,field2]
  Example: WHAT_IF regime_shift to=STRESS asset=silver show=[delta,vega,price]

EXPLAIN regime=<regime> features=[feature1,feature2]
  Example: EXPLAIN regime=TRANSITION features=[ricci_curvature,mst_stress]

SURFACE vol asset=<asset> regime=<regime> strikes=[k1,k2,k3] T=<time>
  Example: SURFACE vol asset=silver regime=STABLE strikes=[0.9,1.0,1.1] T=30d

VOLATILITY SPECIFICATIONS:
- Float: 0.2, 0.25, etc.
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

        # COVARIANCE example
        ("Get the covariance matrix for silver, gold, and DXY in stress regime over 60 days",
         "COVARIANCE assets=[silver,gold,dxy] regime=STRESS window=60d"),

        # TRANSITION example
        ("Show the transition matrix for silver regimes",
         "TRANSITION matrix asset=silver normalize=true"),

        # WHAT_IF example
        ("Analyze how the Greeks would change if silver shifted to a stress regime. Show delta, vega, and price.",
         "WHAT_IF regime_shift to=STRESS asset=silver show=[delta,vega,price]"),

        # EXPLAIN example
        ("Explain the characteristics of transition regimes using Ricci curvature and MST stress",
         "EXPLAIN regime=TRANSITION features=[ricci_curvature,mst_stress]"),

        # SURFACE example
        ("Generate a volatility surface for silver across different strikes (0.9 to 1.1) in stable regime, 30 days",
         "SURFACE vol asset=silver regime=STABLE strikes=[0.9,0.95,1.0,1.05,1.1] T=30d"),

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
