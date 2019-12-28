#Author: Yavuz Faruk Bakman
#Date: 15/08/2019
#Description: Tiny Yolo V2 implementation by Knet framework.
#Currently, it uses pre-trained weigths and doesn't train the model

#Load necessary packages
using Pkg; for p in ("Knet","Random","Glob","FileIO","DelimitedFiles","OffsetArrays","Images",
"ImageDraw","ImageMagick","ImageFiltering","ImageTransformations","Colors","FreeTypeAbstraction","QuartzImageIO","LightXML","ArgParse");
haskey(Pkg.installed(),p) || Pkg.add(p); end

using Knet
using Random, Glob, FileIO, DelimitedFiles, OffsetArrays
using Images, ImageDraw, ImageFiltering, ImageTransformations, Colors
using FreeTypeAbstraction
using IterTools
using LightXML
using ImageMagick,ArgParse
using Statistics

# 2 dictionaries to access number<->class by O(1)
namesdic = Dict("aeroplane"=>1,"bicycle"=>2,"bird"=>3, "boat"=>4,
            "bottle"=>5,"bus"=>6,"car"=>7,"cat"=>8,"chair"=>9,
            "cow"=>10,"diningtable"=>11,"dog"=>12,"horse"=>13,"motorbike"=>14,
            "person"=>15,"pottedplant"=>16,"sheep"=>17,"sofa"=>18,"train"=>19,"tvmonitor"=>20)
numsdic =  Dict(1=>"aeroplane",2=>"bicycle",3=>"bird", 4=>"boat",
            5=>"bottle",6=>"bus",7=>"car",8=>"cat",9=>"chair",
            10=>"cow",11=>"diningtable",12=>"dog",13=>"horse",14=>"motorbike",
            15=>"person",16=>"pottedplant",17=>"sheep",18=>"sofa",19=>"train",20=>"tvmonitor")

MOMENTUM = 0.9
MINIBATCH_SIZE = 32
WEIGHTS_FILE = "yolov2-tiny-voc.weights"
typearr = (Knet.gpu()>=0 ? Knet.KnetArray{Float32} : Array{Float32})
object_scale = Float32(5)
noobject_scale= Float32(1)
class_scale = Float32(1)
coord_scale = Float32(1)
face = newface("DroidSansMono.ttf") #Font type
EXAMPLE_INPUT = "example.jpg"##will be changed
GPU = Knet.gpu()>=0


ANCHORS = typearr(Array{Float32,1}([1.08, 1.19,  3.42, 4.41,  6.63, 11.38,  9.42, 5.11,  16.62, 10.52]))
anchors = [(1.08,1.19),  (3.42,4.41),  (6.63,11.38),  (9.42,5.11),  (16.62,10.52)]

mutable struct Conv; w; stride; padding; f; bnM; gama_beta; end #Define convolutional layer
mutable struct ConvLast; w; b; stride; padding; f; end #Define convolutional layer
mutable struct YoloPad; w; end #Define Yolo padding layer (assymetric padding).
struct Pool; size; stride; pad; end # Define pool layer
#struct Batchnorm2d; depth; end # Kendisi trainliyor zaten autograd value gelince , otherwise normal davranıyor.

include("preprocess.jl")
include("postprocess.jl")
include("loadweights.jl")
include("YoloLoss.jl")

ACC_INPUT = "VOCdevkit/VOC2007/JPEGImages" #Input directory for accuracy calculation
ACC_OUT =   "VOCdevkit/VOC2007/Annotations"

#Define sigmoid function
sigmoid(x) = Float32(1.0) / (Float32(1.0) .+ exp(-x))

#Define Chain
mutable struct Chain
    layers
    Chain(layers...) = new(layers)
end


YoloPad(w1::Int,w2::Int,cx::Int,cy::Int) = YoloPad(typearr(zeros(Float32,w1,w2,cx,cy)))#Constructor for Yolopad
Conv(w1::Int,w2::Int,cx::Int,cy::Int,st,pd,f) = Conv(randn(Float32,w1,w2,cx,cy),st,pd,f,bnmoments(;momentum = MOMENTUM,mean = zeros(Float32,cy),var = ones(Float32,cy)),bnparams(Float32,cy))#Constructor for convolutional layer without bias
ConvLast(w1::Int,w2::Int,cx::Int,cy::Int,st,pd,f) = ConvLast(Param(typearr(randn(Float32,w1,w2,cx,cy)/(13*13))),Param(typearr(randn(Float32,1,1,cy,1)/(13*13))),st,pd,f) # Convolution layer for the last one


#Assymetric padding function
function(y::YoloPad)(x)
    x = reshape(x,14,14,1,:)
    return reshape(conv4(y.w,x; stride = 1),13,13,512,:)
end


(p::Pool)(x) = pool(x; window = p.size, stride = p.stride, padding=p.pad) #pool function
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x) #chain function

(c::Conv)(x)= c.f.(batchnorm(conv4(c.w,x; stride = c.stride, padding = c.padding),c.bnM,c.gama_beta))
(c::ConvLast)(x) = c.f.(conv4(c.w,x; stride = c.stride, padding = c.padding) .+ c.b) #convolutional layer function

#leaky function
function leaky(x)
    return max(Float32(0.1)*x,x)
end

square(x) = x * x

#Tiny Yolo V2 model configuration
model = Chain(Conv(3,3,3,16,1,1,leaky),
              Pool(2,2,0),
              Conv(3,3,16,32,1,1,leaky),
              Pool(2,2,0),
              Conv(3,3,32,64,1,1,leaky),
              Pool(2,2,0),
              Conv(3,3,64,128,1,1,leaky),
              Pool(2,2,0),
              Conv(3,3,128,256,1,1,leaky),
              Pool(2,2,0),
              Conv(3,3,256,512,1,1,leaky),
              Pool(2,1,1),
              YoloPad(2,2,1,1),
              Conv(3,3,512,1024,1,1,leaky),
              Conv(3,3,1024,1024,1,1,leaky),
              ConvLast(1,1,1024,125,1,0,identity))


(m::Chain)(x, truth) = yololoss(truth,m(x))#Array{Float32}(m(x))


function main(args=ARGS)
    s = ArgParseSettings()
    s.description="YoloGit.jl Yavuz Bakman,2019. Trainable Tiny Yolo V2 implementation by Knet framework"
    s.exc_handler=ArgParse.debug_handler
    @add_arg_table s begin
        ("--iou"; arg_type=Float32; default=Float32(0.3); help="If two predictions overlap more than this threshold, one of them is removed")
        ("--iouth"; arg_type=Float32; default=Float32(0.5); help="The threshold for accuracy calculation. If prediction and ground truth overlap more than this threshold, the prediction is counted as true positive, otherwise false positive")
        ("--confth"; arg_type=Float32; default=Float32(0.3); help="The threshold for confidence score. If one prediction's score is more than this threshold, It is taken")
        ("--record"; arg_type= Bool; default=false; help="Sets whether output is saved")
        ("--epochs"; arg_type= Int64; default=Int64(20); help="Number of epochs in training")
        ("--batch_size"; arg_type= Int64; default=Int64(32); help="batch_size for training")
        ("--choose"; arg_type= Bool; default=false; help="Choose which pre-trained model is used")
        ("--example"; arg_type= String; default="Dog.jpg"; help="Example image to display")
    end
    isa(args, AbstractString) && (args=split(args))
    if in("--help", args) || in("-h", args)
       ArgParse.show_help(s; exit_when_done=false)
       return
    end
    println(s.description)
    #Load pre-trained weights into the model
    o = parse_args(args[2:end], s; as_symbols=true)
    if in("accuracy", args)
        if o[:choose]
            global model =  Knet.load("trained1.jld2","model")
        else
            global model =  Knet.load("trained.jld2","model")
        end
        MINIBATCH_SIZE = 1
        images,labels = inputandlabelsdir(ACC_OUT,ACC_INPUT)
        inp,out,imgs = prepareinputlabels(images,labels)
        print("input for accuracy:  ")
        println(summary(inp))
        #Minibatching process
        accdata = minibatch(inp,out,MINIBATCH_SIZE;xtype = typearr)
        AP = accuracy(model,accdata,0.0,o[:iou],o[:iouth])
        display(AP)
        print("Mean average precision: ")
        println(calculatemean(AP))
        if o[:record] == true
            drawdata = minibatch(inp,imgs,MINIBATCH_SIZE; xtype = xtype)
            #output of Voc dataset.
            #return output as [ [(x,y,width,height,classNumber,confidenceScore),(x,y,width,height,classNumber,confidenceScore)...] ,[(x,y,width,height,classNumber,confidenceScore),(x,y,width,height,classNumber,confidenceScore)..],...]
            #save the output images into given location
            result = saveoutput(model,drawdata,o[:confth],o[:iou]; record = true, location = "VocResult")
        end
    end
    if in("loadDisplay", args)
        #Display one test image
        MINIBATCH_SIZE = 1
        EXAMPLE_INPUT = o[:example]
        if o[:choose]
            global model =  Knet.load("trained1.jld2","model")
        else
            global model =  Knet.load("trained.jld2","model")
        end
        displaytest(EXAMPLE_INPUT,model; record = o[:record])
    end
    if in("train", args)
        #prepare data for saving process
        MINIBATCH_SIZE = o[:batch_size]
        f = open(WEIGHTS_FILE)
        getweights(model,f)
        ##Train
        images,labels = inputandlabelsdir(ACC_OUT,ACC_INPUT)
        inp,out,imgs = prepareinputlabels(images,labels)
        print("input for Train:  ")
        println(summary(inp))
        y_batch,b_batch =prepbatches(out)
        y_batch = reshape(y_batch,13*13*5*25,:)
        b_batch = reshape(b_batch,50*4,:)
        total_batch = vcat(b_batch,y_batch)
        dtrn = minibatch(inp,total_batch,MINIBATCH_SIZE; xtype = typearr, ytype = typearr, shuffle=true,partial = true)
        optimizer = adam(model,ncycle(dtrn,o[:epochs]);lr=0.5e-4, beta1=0.9, beta2=0.999, eps=1e-8)
        progress!(optimizer)
        Knet.save("trained1.jld2","model",model)
    end
end

PROGRAM_FILE == "YoloTrain.jl" && main(ARGS)
