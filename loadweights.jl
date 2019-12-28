#Author: Yavuz Faruk Bakman
#Date: 15/08/2019

#Flips given kernel
flipkernel(x) = x[end:-1:1, end:-1:1, :, :]

#loads layers' weights from given file
function getweights(model, file)
    println("Loading weights")
    readconstants!(file)
    #First Conv layer
    loadconv!(model.layers[1],file,3,3,3,16)
    #Second Conv layer
    loadconv!(model.layers[3],file,3,3,16,32)
    #Third Conv layer
    loadconv!(model.layers[5],file,3,3,32,64)
    #4th Conv layer
    loadconv!(model.layers[7],file,3,3,64,128)
    #5th Conv layer
    loadconv!(model.layers[9],file,3,3,128,256)
    #6th Conv layer
    loadconv!(model.layers[11],file,3,3,256,512)
    #YoloPad
    model.layers[13].w[1,1,1,1] = 1
    #7th Conv layer
    loadconv!(model.layers[14],file,3,3,512,1024)
    #8th Conv layer
    loadconv!(model.layers[15],file,3,3,1024,1024)
    println("Weights loaded")
end

#loads the file to given convolutional layer and updates it by batch-normalization
function loadconv!(c,file,d1,d2,d3,d4)
    read!(file, c.gama_beta) #flip gama and beta
    println(summary(c.gama_beta))
    gama = c.gama_beta[d4+1:end]
    c.gama_beta[d4+1:end] = c.gama_beta[1:d4]#set beta
    c.gama_beta[1:d4] = gama #set gama
    mean = Array{Float32}(UndefInitializer(), d4);
    var =  Array{Float32}(UndefInitializer(), d4);
    read!(file,mean)
    read!(file,var)
    c.bnM.mean= reshape(mean,1,1,d4,1)
    c.bnM.var= reshape(var,1,1,d4,1)
    toRead = Array{Float32}(UndefInitializer(), d4*d3*d2*d1);
    read!(file,toRead)
    toRead = reshape(toRead,d1,d2,d3,d4)
    c.w = permutedims(toRead,[2,1,3,4])
    c.w = flipkernel(c.w)
    if gpu() >= 0
        c.w = KnetArray(c.w)
        c.gama_beta = KnetArray(c.gama_beta)
        c.bnM.mean = KnetArray(c.bnM.mean)
        c.bnM.var = KnetArray(c.bnM.var)
    end
    #c.w = Param(c.w) Freeze weights
    #c.gama_beta = Param(c.gama_beta) Freeze weights
end

#read constant and unnecessary numbers from the file
function readconstants!(file)
    major  = read(file,Int32)
    minor = read(file,Int32)
    revision = read(file,Int32)
    iseen = read(file,Int32)#if you use not-voc make it int64
end
