"""
Features:
- include:
In a list "__include:<path>:attr": load path content into list  (insert)
In an object: {"prop": "__include:path:attr"}
global namespace: {"__includefiles": {"sq": "..."}}
"""


class JSONConfigHierarchyLoader:
    def __init__(self, global_namespaces):
        self.global_namespaces = global_namespaces

    def process_document(self, doc):
        return self.process_config_object(doc, {})[0]

    def process_config_object(self, obj, namespaces):
        if isinstance(obj, list):
            processed_list = []
            for item in obj:
                processed_items, is_multiple_items = self.process_config_object(item, namespaces)
                if is_multiple_items:
                    processed_list.extend(processed_items)
                else:
                    processed_list.append(processed_items)
            return processed_list, False
        elif isinstance(obj, dict):
            processed_dict = {}
            include_directives = obj.get("__include")
            if include_directives is not None:
                if not isinstance(include_directives, list):
                    result, _ = self.load_include_directive(include_directives, namespaces)
                    processed_dict.update(result)
                else:
                    for directive in include_directives:
                        result, _ = self.load_include_directive(directive, namespaces)
                        processed_dict.update(result)

            for key, item in obj.items():
                if key == "__include":
                    continue
                processed_items, _ = self.process_config_object(item, namespaces)  # We treat multiple items as just atomic items in dict_attributes
                processed_dict[key] = processed_items
            return processed_dict, False
        elif isinstance(obj, str):
            if obj.startswith("__include:") or obj.startswith(":"):
                prefix, directive = obj.split(":", 1)
                return self.load_include_directive(directive, namespaces)
            else:
                return obj, False
        else:
            return obj, False

    def load_include_directive(self, directive, namespaces):
        parts = directive.rsplit(":")
        namespace = parts[0]

        if len(parts) == 1:
            attribute_path = []
        else:
            attribute_path = parts[1].split()

        namespace_value = self.load_namespace(namespace, namespaces)

        return_value = namespace_value
        for path_elem in attribute_path:
            if isinstance(path_elem, list):
                return_value = return_value[int(path_elem)]
            elif isinstance(path_elem, dict):
                return_value = return_value[path_elem]
            else:
                raise RuntimeError()

        return return_value, True

    def load_namespace(self, namespace_key, local_namespaces):
        local_val = local_namespaces.get(namespace_key)
        if local_val is not None:
            return local_val

        global_val = self.global_namespaces.get(namespace_key)
        if global_val is None:
            raise RuntimeError()

        return global_val


if __name__ == "__main__":
    loader = JSONConfigHierarchyLoader({
        "a": 10,
        "b": [000000],
        "c": {"k2": 10},
        "d": {"k50": 10},
    })

    result = loader.process_document({
        "__include": ["c", "d"],
        "k1": "__include:a",
        "k50": 0
    })

    print(result)
