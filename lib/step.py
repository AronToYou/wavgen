from lib.pyspcm import *


######### Step Class #########
class Step:
    """ NOTE: Indexes start at 0!!
        MEMBER VARIABLES:
            + CurrentStep -- The Sequence index for this step.
            + SegmentIndex - The index into the Segment array for the associated Wave.
            + Loops -------- Number of times the Wave is looped before checking continue Condition.
            + NextStep ----- The Sequence index for the next step.
            -- OPTIONAL --
            + Condition ---- A keyword to indicate: if a trigger is necessary for the step
                            to continue to the next, or if it should be the last step.
                            ['trigger', 'end'] respectively.
                            Defaults to None, meaning the step continues after looping 'Loops' times.

        USER METHODS:

        PRIVATE METHODS:

    """
    Conds = {  # Dictionary of Condition keywords to Register Value Constants
        None      : SPCSEQ_ENDLOOPALWAYS,
        'trigger' : SPCSEQ_ENDLOOPONTRIG,
        'end'     : SPCSEQ_END
    }

    def __init__(self, cur, seg, loops, nxt, cond=None):
        self.CurrentStep = cur
        self.SegmentIndex = seg
        self.Loops = loops
        self.NextStep = nxt
        self.Condition = self.Conds.get(cond)

        assert self.Condition is not None, "Invalid keyword for Condition."
