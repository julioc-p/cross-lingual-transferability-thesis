def validate_sparql_query_result(response):
    if "results" in response and not response["results"]["bindings"]:
        return False

    if "results" in response:
        for binding in response["results"]["bindings"]:
            for var, value in binding.items():
                if (
                    value.get("datatype") == "http://www.w3.org/2001/XMLSchema#integer"
                    and int(value["value"]) == 0
                ):
                    return False

    if "boolean" in response and not response["boolean"]:
        return False

    if "results" not in response and "boolean" not in response:
        return False

    if "results" in response:
        for binding in response["results"]["bindings"]:
            for var, value in binding.items():
                if value["type"] == "literal" and value["value"] == "":
                    return False

    return True
