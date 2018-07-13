import os
import pickle
import base64

import logging
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

#class HierarchicalDict:
#    pass
#
#
#class ComponentRunner:
#    def derive_component_cache_path(self, input_data, component):
#        cache_path_parts = component.derive_cache_path_components()
#        component_dependency_keys = component.component_dependency_keys
#        for component_key in component_dependency_keys:
#            dependency = self.components[component_key]
#            dependency_cache_path_parts = dependency.derive_cache_path_components()
#
#            for key, val in dependency_cache_path_parts:
#                cache_path_parts[component_key + "." + key] = val
#
#        partial_strings = []
#        for key, val in sorted(cache_path_parts.items()):
#            partial_strings.append(key + "=" + val)
#
#        fname = urllib.parse.quote("&".join(partial_strings))
#        path = os.path.join("compcache", input_data.cache_key, fname)
#
#        return path
#
#    def run(self, component, context, *args, **kwargs):
#        should_recompute = True
#        if isinstance(component, CacheableComponent):
#            cache_path = self.derive_component_cache_path(self.input_data, component)
#            if os.path.isfile(cache_path):
#                component_data = component.load_cache_data(cache_path)
#                should_recompute = True
#        if should_recompute:
#            component_data = component.compute(self.input_data, context, *args, **kwargs)
#
#        new_context = component.apply_data_to_context(component_data, context)
#
#        return new_context


class ComponentRegistry:
    registry = {}

    @classmethod
    def find_impl(cls, key):
        return cls.registry[key]


def component(key):
    def impl(cls):
        ComponentRegistry.registry[key] = cls
        return cls

    return impl


class ComponentConfigParser:
    def parse_config(self, config_dict):
        self.parse_partial_config(self.root_schema, config_dict)

    def parse_partial_config(self, root_schema, config_dict):
        for key, val_schema in root_schema.items():
            pass


class DictValue:
    def __init__(self, schema):
        self.schema = schema

    def parse(self, config):
        result_dict = {}
        for key, val_parser in self.schema.items():
            raw_val = config.get(key)
            processed_val = val_parser.parse(raw_val)
            result_dict[key] = processed_val

        for key in set(config) - set(self.schema):
            logger.warn("Unknown key {}".format(key))

        return result_dict


class Value:
    def parse(self, config):
        return config


class ComponentValue:
    def parse(self, config):
        implementation_name = config["__impl"]
        component_cls = ComponentRegistry.find_impl(implementation_name)
        config_value = DictValue(component_cls.get_parameters())
        parameters = config_value.parse(config)

        return component_cls(parameters)


class Component:
    def __init__(self, config):
        pass

    @classmethod
    def get_parameters(self):
        pass


@component("test")
class TestComponent:
    @classmethod
    def get_parameters(self):
        return {
            "b": ComponentValue(),
        }


@component("test2")
class TestComponent2:
    def __init__(self, config):
        pass

    @classmethod
    def get_parameters(self):
        return {
            "a": Value(),
        }



class CacheManager:
    def __init__(self):
        pass

    def key_to_path(self, key):
        return "test-caches/" + str(base64.encodestring(bytearray(key, "utf8")), "utf8")

    def load_cached_data(self, key):
        presumed_cache_path = self.key_to_path(key)
        if os.path.isfile(presumed_cache_path):
            with open(presumed_cache_path, "rb") as f:
                return pickle.load(f)

        return None

    def save_cached_data(self, key, data):
        cache_path = self.key_to_path(key)
        with open(cache_path, "wb") as f:
            return pickle.dump(data, f)


class Promise:
    def __init__(self, parent, identity_key, cache_manager):
        self.cache_manager = cache_manager
        self.parent = parent
        self.identity_key = identity_key

    def compute_full_identity_key(self):
        if self.parent is None:
            return self.identity_key

        if self.identity_key is None:
            return self.parent.compute_full_identity_key()

        parent_key = self.parent.compute_full_identity_key()

        if parent_key is None:
            return None

        return parent_key + "." + self.identity_key

    def chain(self, executable, postprocessor=None, key=None):
        return ComputedPromise(
            self,
            executable,
            postprocessor=postprocessor,
            identity_key=key,
            cache_manager=self.cache_manager
        )


class ComputedPromise(Promise):
    def __init__(self, parent, callable, postprocessor, identity_key, cache_manager):
        super().__init__(parent, identity_key, cache_manager)
        self.cache_manager = cache_manager
        self.callable = callable
        self.postprocessor = postprocessor

    def get(self):
        full_identity_key = self.compute_full_identity_key()
        data = None
        if self.cache_manager:
            data = self.cache_manager.load_cached_data(full_identity_key)
        if data is None or self.postprocessor is not None:
            input_data = self.parent.get()
        if data is None:
            data = self.callable(input_data)

            if self.cache_manager and self.identity_key is not None:
                self.cache_manager.save_cached_data(full_identity_key, data)

        if self.postprocessor is not None:
            result = self.postprocessor(input_data, data)
        else:
            result = data

        return result


class ConstantPromise(Promise):
    def __init__(self, data, identity_key, cache_manager):
        super().__init__(None, identity_key, cache_manager)
        self.data = data

    def get(self):
        return self.data


if __name__ == "__main__":
    base_promise = ConstantPromise(4, identity_key="4", cache_manager=CacheManager())
    print(base_promise.chain(
        lambda x: x + 10
    ).chain(lambda x: x * 10, key="step2").get())


#cluster_promise = input_documents.chain(clusterer.cluster, key=clusterer.identity_key)  # [Cluster(id=..., members=...)]
#
#candidates_promise = cluster_promise.chain_foreach(generate_candidates, match_candidate_to_cluster, key=clusterer.identity_key)   # [Cluster(id=..., candidates=...)]
#global_tr_promise = input_documents.chain(compute_global_tr)  # TR-Dict
#local_tr_promise = cluster_promise.chain_foreach(compute_cluster_tr, lambda cl, res: setattr(cl, "local_tr_scores", res))
#