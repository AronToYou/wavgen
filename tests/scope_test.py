from wavgen import *
import pyvisa

MEM_SIZE = 4_294_967_296  # Board Memory

r = [2.094510589860613, 5.172224588379723, 2.713365750754814, 2.7268654021553975, 1.   /
     9455621726067513, 2.132845902763719, 5.775685169342227, 4.178303582622483, 1.971  /
     4912917733933, 1.218844007759545, 4.207174369712666, 2.6609861484752124, 3.41140  /
     54221128125, 1.0904071328591276, 1.0874359520279866, 1.538248528697041, 0.501676  /
     9726252504, 2.058427862897829, 6.234202186024447, 5.665480185178818]

if __name__ == '__main__':
    ## Setup Scope ##
    rm = pyvisa.ResourceManager()
    for inst in rm.list_resources():
        scope = rm.open_resource(inst)
        try:
            print(scope.query("*IDN?"))
        except:
           print("failed on ", inst)

    # freq_A = [90E6 + j*1E6 for j in range(10)]
    # phases = r[:len(freq_A)]
    #
    # sweep_size = MEM_SIZE // 8
    # assert (sweep_size % 32) == 0, "Not 32 bit aligned."
    #
    # ## First Superposition defined with a list of frequencies ##
    # A = Superposition(freq_A, phases=phases, resolution=int(1E6))
    # A.compute_waveform()
    #
    # ## Setup and Run the Card ##
    # dwCard = Card()
    # dwCard.load_waveforms(A)
    # dwCard.wiggle_output(duration=1000.0, block=False)



