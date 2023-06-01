# first start by reading in the file with state names
state_names_file = "tmp.txt"
flux_names_file = "tmp1.txt"
ode_file = "ampk_MM_single_mech.jl"

f_states = open(state_names_file, "r")

# loop though all lines in the file
# each line is a state + state number pair, so process accordingly
for line in f_states:
    split_line = line.split(' ')

    state_number = split_line[1] + "(t)" # only works because I know the ordering and spacing of things!
    dt_state_number = split_line[1] + "'(t) = "
    state_name = split_line[3].replace("'", "").replace("\n", "")
    state_reference = 'y.'+state_name.replace(":(),","")
    ode_state_reference = "'"+state_name.replace(":(),","")+"':"
    
    print(state_reference)
    print(ode_state_reference)

    # replace state names with state numbers in the fluxes file
    # now read in the file with the fluxes and
    f_fluxes = open(flux_names_file, "rt") # open in read mode
    data = f_fluxes.read()
    data = data.replace(state_reference, state_number)
    f_fluxes.close()
    f_fluxes = open(flux_names_file, "wt") # open in write mode
    # overrite the input file with the resulting data
    f_fluxes.write(data)
    #close the file
    f_fluxes.close()

    # now replace state names in ode file with state numbers
    # now read in the file with the fluxes and
    f_ode = open(ode_file, "rt") # open in read mode
    data = f_ode.read()
    data = data.replace(ode_state_reference, dt_state_number)
    f_ode.close()
    f_ode = open(ode_file, "wt") # open in write mode
    # overrite the input file with the resulting data
    f_ode.write(data)
    #close the file
    f_ode.close()

# replace flux numbers with full flux definitions in the julia file
f_fluxes = open(flux_names_file, "rt").readlines() # open in read mode
for line in reversed(f_fluxes):
    split_line = line.split(' = ')
    if len(split_line) > 1: # ignore lines that don't have an equals sign
        flux_number = split_line[0]
        flux = "("+split_line[1].replace("'", "").replace(" ", "").replace("\n", "")+")"
    
       # now replace flux numbers in ode file with fluxes
        # now read in the file with the fluxes and
        f_ode = open(ode_file, "rt") # open in read mode
        data = f_ode.read()
        data = data.replace(flux_number, flux)
        f_ode.close()
        f_ode = open(ode_file, "wt") # open in write mode
        # overrite the input file with the resulting data
        f_ode.write(data)
        #close the file
        f_ode.close()
