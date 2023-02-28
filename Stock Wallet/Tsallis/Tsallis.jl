module Tsallis

function qlog(x, q)
    if q==1
        return log(x)
    else
        return (x^(1. -q) - 1.) / (1. -q)
    end
end

function qexp(x, q)
    if q==1
        return exp(x)
    else
        return max(0., (1. + (1. - q)*x)^(1/(1. - q)))
    end
end

function qproduct(x, y, q)
    return max(0., (x^(1. - q) + y^(1. - q) - 1.)^(1/(1. - q)))
end

function qsum(x, y, q)
    return x + y + (1. - q)*x*y
end

function qsubtract(x, y, q)
    return 
end

export log
export exp
export product
export qsum
end


module qGaussian

using Statistics


export distribution

end