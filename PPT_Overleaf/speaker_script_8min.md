# FinSight 8-Minute Speaker Script

Suggested pace: about 7 minutes 20 seconds to 7 minutes 50 seconds for a slower, clear English delivery. Use the remaining time for slide transitions and short pauses.

## Slide 1: Title

Good morning everyone. We are Group 4.2, and our project is called FinSight. It is an evidence-first financial analysis chatbot for ARIN7012. In this presentation, we will explain the background, our system goal, the main backend modules, current performance, a real demo, and future work.

## Slide 2: Financial chatbot answers must be grounded

Financial chatbot answers are different from casual chatbot answers. A financial answer may influence a user's buy, hold, or sell decision. The user's real intent is often hidden in a short question, such as "What do you think about Ping An?" To answer safely, the system must understand the intent, find evidence, and explain limitations.

## Slide 3: The project goal is an evidence-first financial chatbot

Our goal is not to directly give investment advice. Instead, FinSight first understands the question, retrieves reliable evidence, analyzes market and text signals, and then prepares structured JSON for the chatbot response. This makes the final answer more traceable, more controlled, and easier to evaluate.

## Slide 4: FinSight is powered by the ARIN intelligence backend

This is the overall architecture. The user sends a query through the chatbot UI. The backend API calls the NLU module, then retrieval, then answer handoff. The key point is that every stage produces structured artifacts, such as `nlu_result`, `retrieval_result`, and final answer JSON. So the frontend does not need to guess the intent again.

## Slide 5: NLU turns raw language into source requirements

The first core module is NLU. It normalizes Chinese and English queries, resolves entities, predicts product type, intent, topic, and question style. It also detects missing information and risk flags. The final output is a source plan, telling retrieval which evidence sources should be used.

## Slide 6: Retrieval gathers ranked evidence from documents and APIs

The retrieval module follows the NLU source plan. It collects documents such as news, announcements, reports, and FAQs. It also collects structured data from market and fundamental providers. Then it ranks and packages the evidence, including coverage information, warnings, and traceable evidence IDs.

## Slide 7: Market analysis summarizes signals without giving advice

After retrieval, the market analyzer summarizes numerical signals. For example, it can compute returns, moving averages, RSI, MACD, volatility, valuation, and macro signals. But it does not make a final buy or sell decision. It only prepares evidence and data readiness information for downstream response generation.

## Slide 8: Sentiment analysis reuses retrieved evidence and entity context

The sentiment module runs after NLU and retrieval. It does not build another understanding system. Instead, it reuses retrieved documents and entity context. It filters irrelevant documents, checks language and source type, and then classifies finance-related text as positive, negative, or neutral.

## Slide 9: Answer handoff gives the chatbot validated JSON

The answer handoff module prepares the final chatbot response format. It removes raw debug information, keeps compact evidence, and validates the output JSON. The answer includes key points, limitations, evidence IDs, a disclaimer, and predicted follow-up questions. This helps the frontend show a safer and cleaner answer.

## Slide 10: Current results support the demo story

Here are our current evaluation results. Finance recall is 0.9881, out-of-domain rejection is 0.9750, intent F1 is 0.8727, topic F1 is 0.9474, and retrieval NDCG at 10 is 0.8849. We also passed a large bilingual evaluation, fuzz tests, and a focused test run. These results support the reliability of the demo.

## Slide 11: Demo English finance query

This is a real demo screenshot. The user asks, "How is Ping An Insurance?" The system identifies the company, retrieves market price, fundamentals, industry valuation, and related news. The answer also lists evidence sources. This shows that the chatbot response is grounded in structured evidence, not just generated from memory.

## Slide 12: Demo Chinese market query

This screenshot shows a Chinese query: "茅台股票今天涨了吗?" The system returns the latest price movement, intraday high and low, and technical signals such as moving averages and RSI. It also includes the evidence source and a risk disclaimer. This shows that the same pipeline works for Chinese financial questions.

## Slide 13: Demo out-of-scope query

This third demo is important for safety. The user asks, "Write me a poem." This is not a financial question. The system rejects it as out of scope and says that no financial evidence is available. This prevents the chatbot from producing unsupported answers outside its designed domain.

## Slide 14: Future work focuses on integration and robustness

For future work, we will first merge the chatbot frontend with the backend evidence contract. We also want to expand market coverage beyond the current China-market version. Another direction is improving live-provider monitoring and fallback behavior, so the system remains stable when external data sources fail.

## Slide 15: Conclusion

To conclude, FinSight separates query understanding, evidence retrieval, analysis, and answer generation. The core backend is classical, explainable, and measurable. The chatbot receives traceable JSON instead of ungrounded guesses. This makes financial chatbot answers safer, more transparent, and easier to improve in the future. Thank you.
