using StructuralIdentifiability

file_list = ["../odes/ampk_MA_double_mech.jl", "../odes/ampk_MA_single_mech.jl", "../odes/ampk_MM_double_mech.jl", "../odes/ampk_MM_single_mech.jl", "../odes/ampk_newmech_MA_single.jl", "../odes/ampk_qss_single.jl"]
model_list = ["MA_double", "MA_single", "MM_double", "MM_single", "newmech_MA_single", "qss_single"]
params_to_test = [
    ["kOffAMP", "kOffADP", "kOffATP", "kOnCaMKK", "kPhosCaMKK", "kOnLKB1", "kPhosLKB1", "kOnPP", "kDephosPP", "kOnAMPK", "kPhosAMPK", "kOnPP1", "kDephosPP1"],
    ["kOffAMP", "kOffADP", "kOffATP", "kOnCaMKK", "kPhosCaMKK", "kOnLKB1", "kPhosLKB1", "kOnPP", "kDephosPP", "kOnAMPK", "kPhosAMPK", "kOnPP1", "kDephosPP1"],
    ["kOffAMP", "kOffADP", "kOffATP", "kCaMKK", "KmCaMKK", "kLKB1", "KmLKB1", "kPP", "KmPP", "kAMPK", "KmAMPK", "kPP1", "KmPP1"], 
    ["kOffAMP", "kOffADP", "kOffATP", "kCaMKK", "KmCaMKK", "kLKB1", "KmLKB1", "kPP", "KmPP", "kAMPK", "KmAMPK", "kPP1", "KmPP1"], 
    ["kOffAMP", "kOffADP", "kOffATP", "kOnCaMKK", "kPhosCaMKK", "kOnLKB1", "kPhosLKB1", "kOnPP", "kDephosPP", "kOnAMPK", "kPhosAMPK", "kOnPP1", "kDephosPP1", "beta"],
    ["kOffAMP", "kOffADP", "kOffATP", "kCaMKK", "KmCaMKK", "kLKB1", "KmLKB1", "kPP", "KmPP", "kAMPK", "KmAMPK", "kPP1", "KmPP1", "beta"]
]

for (file, model, plist) in zip(file_list, model_list, params_to_test)
    println("Running: $model...")
    include(file)

    local_id = assess_local_identifiability(ode, 0.99, plist)

    # save to file for this model
    # create file to save results
    file = open("../identifiability/local_ID_$model.txt", "w")
    println(file, "Locally Identifiable parameters:")
    for (key, value) in local_id
        if value == 1
            print(file, "$key, ")
        end
    end
    println(file, "")
    println(file, "Nonidentifiable parameters:")
    for (key, value) in local_id
        if value == 0
            print(file, "$key, ")
        end
    end
    close(file)
end