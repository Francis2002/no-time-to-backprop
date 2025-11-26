def print_tree_keys(tree):
        # ASCII tree printer using '|' and '-' to visualize structure
        def shape_info(x):
            try:
                s = getattr(x, "shape", None)
                if s is not None and len(tuple(s)) > 0:
                    return f"  shape={tuple(s)}"
            except Exception:
                pass
            return ""

        def is_mapping(x):
            try:
                return isinstance(x, dict) or hasattr(x, "items")
            except Exception:
                return False

        def iter_children(x):
            if is_mapping(x):
                try:
                    return [(str(k), v) for k, v in x.items()]
                except Exception:
                    return []
            if isinstance(x, (list, tuple)):
                return [(f"[{i}]", v) for i, v in enumerate(x)]
            return []

        def rec(x, prefix=""):
            children = iter_children(x)
            n = len(children)
            if n == 0:
                # leaf at root
                print(prefix + "- leaf" + shape_info(x))
                return

            for i, (label, child) in enumerate(children):
                last = (i == n - 1)
                connector = "|- "
                line = prefix + connector + f"{label}"
                if is_mapping(child) or isinstance(child, (list, tuple)):
                    print(line)
                    rec(child, prefix + ("   " if last else "|  "))
                else:
                    print(line + shape_info(child))

        rec(tree, "")