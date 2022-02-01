from contextlib import contextmanager
from time import perf_counter
import pandas as pd

@contextmanager
def perf_count(name: str, enable: bool = True):
    """
    Prints wall time for the code block to STDOUT

    Example:
        with perf_count('test code'):
            sleep(10)
        # Writes to stdout:
        # {name}: {latency}s
    """
    if not enable:
        yield
    else:
        s = perf_counter()
        yield
        t = perf_counter()
        d = t-s
        if d < 1:
            print('{0}: {1:.2f}ms'.format(name, d*1e3))
        else:
            print('{0}: {1:.2f}s'.format(name, d))

def explain_to_str(df: pd.DataFrame, mode: str = 'simple', extended: bool = None):
    """
    capture output of dataframe's explain() to write it to a file

    :param extended: boolean, default ``False``. If ``False``, prints only the physical plan.
    When this is a string without specifying the ``mode``, it works as the mode is
    specified.

    :param mode: specifies the expected output format of plans.

            * ``simple``: Print only a physical plan.
            * ``extended``: Print both logical and physical plans.
            * ``codegen``: Print a physical plan and generated codes if they are available.
            * ``cost``: Print a logical plan and statistics if they are available.
            * ``formatted``: Split explain output into two sections: a physical plan outline \
                    and node details.
    """
    basestring = unicode = str
    if extended is not None and mode is not None:
        raise Exception("extended and mode should not be set together.")

    # For the no argument case: df.explain()
    is_no_argument = extended is None and mode is None

    # For the cases below:
    #   explain(True)
    #   explain(extended=False)
    is_extended_case = isinstance(extended, bool) and mode is None

    # For the case when extended is mode:
    #   df.explain("formatted")
    is_extended_as_mode = isinstance(extended, basestring) and mode is None

    # For the mode specified:
    #   df.explain(mode="formatted")
    is_mode_case = extended is None and isinstance(mode, basestring)

    if not (is_no_argument or is_extended_case or is_extended_as_mode or is_mode_case):
        argtypes = [
            str(type(arg)) for arg in [extended, mode] if arg is not None]
        raise TypeError(
            "extended (optional) and mode (optional) should be a string "
            "and bool; however, got [%s]." % ", ".join(argtypes))

    # Sets an explain mode depending on a given argument
    if is_no_argument:
        explain_mode = "simple"
    elif is_extended_case:
        explain_mode = "extended" if extended else "simple"
    elif is_mode_case:
        explain_mode = mode
    elif is_extended_as_mode:
        explain_mode = extended

    return df._sc._jvm.PythonSQLUtils.explainString(
        df._jdf.queryExecution(), explain_mode
    )
