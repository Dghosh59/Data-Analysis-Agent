import contextlib
import matplotlib.pyplot as plt
import io
import traceback
import pandas as pd
def execute_generated_code(code: str):
    output = io.StringIO()
    local_vars = {}
    fig_before = plt.get_fignums()
    try:
        with contextlib.redirect_stdout(output):
            exec(code, {"plt": plt, "pd": pd}, local_vars)
        output_text = output.getvalue()
        fig_after = plt.get_fignums()
        if len(fig_after) > len(fig_before):
            return output_text, plt.gcf(), None
        return output_text, None, None
    except Exception as e:
        return "", None, traceback.format_exc()
