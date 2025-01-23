def _format_function_definitions(tools):
    """
    Formats tool definitions in the format that OpenAI appears to use.
    https://github.com/forestwanglin/openai-java/blob/main/jtokkit/src/main/java/xyz/felh/openai/jtokkit/utils/TikTokenUtils.java
    :param tools:
    :return:
    """
    lines = ["namespace functions {", ""]
    for tool in tools:
        function = tool.get("function")
        if function_description := function.get("description"):
            lines.append(f"// {function_description}")
        function_name = function.get("name")
        parameters = function.get("parameters", {})
        properties = parameters.get("properties")
        if properties and properties.keys():
            lines.append(f"type {function_name} = (_: {{")
            lines.append(_format_object_parameters(parameters, 0))
            lines.append("}) => any;")
        else:
            lines.append(f"type {function_name} = () => any;")
        lines.append("")
    lines.append("} // namespace functions")
    return "\n".join(lines)


def _format_object_parameters(parameters, indent):
    properties = parameters.get("properties")
    if not properties:
        return ""
    required_params = parameters.get("required", [])
    lines = []
    for key, props in properties.items():
        description = props.get("description")
        if description:
            lines.append(f"// {description}")
        question = "?"
        if required_params and key in required_params:
            question = ""
        lines.append(f"{key}{question}: {_format_type(props, indent)},")
    return "\n".join([" " * max(0, indent) + line for line in lines])


def _format_type(props, indent):
    _type = props.get("type")
    if _type == "string":
        return " | ".join([f'"{item}"' for item in props.get("enum", [])]) or "string"
    if _type == "array":
        return f"{_format_type(props['items'], indent)}[]"
    if _type == "object":
        return f"{{\n{_format_object_parameters(props, indent + 2)}\n}}"
    if _type in {"integer", "number"}:
        return " | ".join([f'"{item}"' for item in props.get("enum", [])]) or "number"
    if _type in {"boolean", "null"}:
        return _type
    return "any"
