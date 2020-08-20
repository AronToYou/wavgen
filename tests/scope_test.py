import pyvisa

## Setup Scope ##
rm = pyvisa.ResourceManager()
for inst in rm.list_resources():
    scope = rm.open_resource(inst)
    try:
        print(scope.query("*IDN?"))
    except Exception as err:
        print("failed on ", inst)
        print(type(err), err, '\n')
    scope.close()
rm.close()
