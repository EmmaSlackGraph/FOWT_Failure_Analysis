class FMEAMode:

    def __init__(self, num = None, fail_name="", effects=[], causes=[], multi=False, system=""):
        self.num = num
        self.name = fail_name
        self.effects = effects
        self.pcauses = causes
        self.multi = multi
        self.system = system
        