def validate_sparql_query_result(response):
    # Detect SELECT queries with no matches
    if "results" in response and not response["results"]["bindings"]:
        return False

    # Detect COUNT queries returning 0
    if "results" in response:
        for binding in response["results"]["bindings"]:
            for var, value in binding.items():
                if (
                    value.get("datatype") == "http://www.w3.org/2001/XMLSchema#integer"
                    and int(value["value"]) == 0
                ):
                    return False

    # Detect ASK queries returning false
    if "boolean" in response and not response["boolean"]:
        return False

    # Detect empty RDF graph for DESCRIBE/CONSTRUCT
    if "results" not in response and "boolean" not in response:
        # Assume RDF graph, needs manual inspection for emptiness
        return False

    # Detect empty literals
    if "results" in response:
        for binding in response["results"]["bindings"]:
            for var, value in binding.items():
                if value["type"] == "literal" and value["value"] == "":
                    return False

    return True
