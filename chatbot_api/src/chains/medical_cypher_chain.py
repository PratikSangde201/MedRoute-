class _DummyCypherChain:
    def invoke(self, payload):
        query = payload.get("query", "") if isinstance(payload, dict) else str(payload)
        return {"result": "", "query": query}


chatbot_cypher_chain = _DummyCypherChain()
