let image = document.getElementById("pic");
let div = document.getElementById("div");

let canvas1 = document.createElement('canvas');
canvas1.setAttribute("width", image.width.toString());
canvas1.setAttribute("height", image.height.toString());
canvas1.setAttribute("id", "canvas1");

div.appendChild(canvas1);

let canvas2 = document.createElement("canvas");
canvas2.setAttribute("width", image.width.toString());
canvas2.setAttribute("height", image.height.toString());
canvas2.setAttribute("id", "canvas2");

div.appendChild(canvas2);

let canvas = document.getElementById("canvas1");
let ctx = canvas.getContext("2d");

let sourceImage = new Image(image.width, image.height);
let imageData = null;

sourceImage.onload = function()
{
    ctx.drawImage(sourceImage, 0, 0);
    imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

    let trueGreyscale = ctx.createImageData(image.width, image.height);

    for(let i = 0; i < imageData.data.length; i += 4)
    {
        let color = imageData.data[i] * 0.21 + imageData.data[i + 1] * 0.72 + imageData.data[i + 2] * 0.07;

        trueGreyscale.data[i] = color;
        trueGreyscale.data[i + 1] = color;
        trueGreyscale.data[i + 2] = color;
        trueGreyscale.data[i + 3] = 255;
    }
    let tempctx = document.getElementById("canvas2").getContext("2d");
    tempctx.putImageData(trueGreyscale, 0, 0);

    execute();
}
sourceImage.src = image.src;

function execute()
{
    let population = 40;
    let netList = [];
    for(let i = 0; i < population; ++i)
    {
        netList.push(new MLP(3, 4, 1));
    }

    let maxIterations = 300;

    for(let iteration = 0; iteration < maxIterations; ++iteration)
    {
        let fitnesses = [];
        for(let i = 0; i < population; ++i)
        {
            let fitness = 0;
            for(let j = 0; j < 30; ++j)
            {
                let red = getRand(0, 255);
                let green = getRand(0, 255);
                let blue = getRand(0, 255);
                netList[i].input([red / 255 * 2 - 1, green / 255 * 2 - 1, blue  / 255 * 2 - 1]);

                let expectedOutput = (red * 0.21 + green * 0.72 + blue * 0.07) / 255;
                let output = netList[i].getOutput()[0];

                let score = 0;
                let difference = Math.abs(expectedOutput - output);
                if(difference >= 0.05)
                {
                    score = 0;
                }
                else
                {
                    score = 0.05 - difference;
                }

                fitness += score;
            }
            fitnesses.push(fitness);
        }

        sortNeuralNets(netList, fitnesses);
        if(iteration === maxIterations - 1)
        {
            continue;
        }

        let succsessRates = getSuccessRates(fitnesses);
        netList = getNextGen(netList, succsessRates);
    }

    let proccessedImage = ctx.createImageData(image.width, image.height);
    for(let i = 0; i < imageData.data.length; i += 4)
    {
        netList[39].input([imageData.data[i] / 255 * 2 - 1, imageData.data[i + 1] / 255 * 2 - 1, imageData.data[i + 2] / 255 * 2 - 1]);
        let value = netList[39].getOutput()[0];
        let color = 0;

        if(value > 1)
        {
            color = 255;
        }
        else if(value > 0)
        {
            color = 255 * value;
        }

        proccessedImage.data[i] = color;
        proccessedImage.data[i+1] = color;
        proccessedImage.data[i+2] = color;
        proccessedImage.data[i+3] = 255;
    }

    ctx.putImageData(proccessedImage, 0, 0);
}
