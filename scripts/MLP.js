function getRand(min = 0, max = 1)
{
    return min + (Math.random() * (max - min));
}

function MLP(nInputs, nHidden, nOutputs, chromosome = null)
{
    this.nInputs = nInputs;
    this.nHidden = nHidden;
    this.nOutputs = nOutputs;
    this.hiddenL = [];
    this.outputL =[];
    this.chromosome = [];
    this.outputResults = [];

    this.defaultConstructor = function()
    {
        for(let i = 0; i < nHidden; ++i)
        {
            this.hiddenL.push([]);
            for(let j = 0; j <= nInputs; ++j)
            {
                let weight = getRand(-1, 1);
                this.hiddenL[i].push(weight);
                this.chromosome.push(weight);
            }
        }

        for(let i = 0; i < nOutputs; ++i)
        {
            this.outputL.push([]);
            for(let j = 0; j <= nHidden; ++j)
            {
                let weight = getRand(-1, 1);
                this.outputL[i].push(weight);
                this.chromosome.push(weight);
            }
        }
    }

    this.geneConstructor = function(chromosome)
    {
        let arrayPos = 0;
        for(let i = 0; i < nHidden; ++i)
        {
            this.hiddenL.push([]);
            for(let j = 0; j <= nInputs; ++j)
            {
                this.hiddenL[i].push(chromosome[arrayPos]);
                this.chromosome.push(chromosome[arrayPos]);
                arrayPos++;
            }
        }

        for(let i = 0; i < nOutputs; ++i)
        {
            this.outputL.push([]);
            for(let j = 0; j <= nHidden; ++j)
            {
                this.outputL[i].push(chromosome[arrayPos]);
                this.chromosome.push(chromosome[arrayPos]);
                arrayPos++;
            }
        }
    }

    if(chromosome === null)
        this.defaultConstructor();
    else
        this.geneConstructor(chromosome);

    this.hiddenActivationFunc = function(val)
    {
        //return Math.max(0, val);
        return Math.tanh(val);
    }

    this.outputActivationFunc = function(val)
    {
        //return Math.max(0, val);
        return Math.tanh(val);
    }

    this.input = function(input)
    {
        this.outputResults = [];
        let hiddenLResults = [];
        for(let i = 0; i < this.nHidden; ++i)
        {
            let netInput = 0;
            for(let j = 0; j < this.nInputs; ++j)
            {
                netInput += this.hiddenL[i][j] * input[j];
            }

            netInput += this.hiddenL[i][this.nInputs];
            hiddenLResults.push(this.hiddenActivationFunc(netInput));
        }

        for(let i = 0; i < this.nOutputs; ++i)
        {
            let netInput = 0
            for(let j = 0; j < hiddenLResults.length; ++j)
            {
                netInput += this.outputL[i][j] * hiddenLResults[j];
            }

            netInput += this.outputL[i][this.nHidden];
            this.outputResults.push(this.outputActivationFunc(netInput));
        }
    }

    this.getChromosome = function()
    {
        let copy = [];
        for(let i = 0; i < this.chromosome.length; ++i)
        {
            copy.push(this.chromosome[i]);
        }

        return copy;
    }

    this.getOutput = function()
    {
        let copy = [];
        for(let i = 0; i < this.outputResults.length; ++i)
        {
            copy.push(this.outputResults[i]);
        }

        return copy;
    }
}

function breed(parent1, parent2)
{
    let splitIndex = Math.floor(getRand(0, parent1.length));
    let childChromosome = [];
    for(let i = 0; i < splitIndex; ++i)
    {
        childChromosome.push(parent1[i]);
    }

    for(let i = splitIndex; i < parent2.length; ++i)
    {
        childChromosome.push(parent2[i]);
    }

    return childChromosome;
}

function getNextGen(netList, successRate)
{
    let nInputs = netList[0].nInputs;
    let nHidden = netList[0].nHidden;
    let nOutputs = netList[0].nOutputs;
    let newGen = [];

    for(let i = 0; i < netList.length / 2; ++i)
    {
        let parent1 = getRand();
        let parent2 = getRand();

        let index1 = 0;
        for(let j = 0; j < successRate.length; ++j)
        {
            if(parent1 <= successRate[j])
                break;

            index1++;
        }

        let index2 = 0;
        for(let j = 0; j < successRate.length; ++j)
        {
            if(parent2 <= successRate[j])
                break;

            index2++;
        }

        let chromosome1 = netList[index1].getChromosome();
        let chromosome2 = netList[index2].getChromosome();

        let childChromosome1 = breed(chromosome1, chromosome2);
        let childChromosome2 = breed(chromosome1, chromosome2);

        let mutation = getRand(0, 100);
        if(mutation > 98)
        {
            let mutationIndex = Math.floor(getRand(0, childChromosome1.length));
            childChromosome1[mutationIndex] = getRand(-1, 1);

            mutationIndex = Math.floor(getRand(0, childChromosome2.length));
            childChromosome2[mutationIndex] = getRand(-1, 1);
        }

        mutation = getRand(0, 100);
        if(mutation > 95)
        {
            let mutationIndex = Math.floor(getRand(0, childChromosome1.length));
            childChromosome1[mutationIndex] += getRand(-0.005, 0.005);

            mutationIndex = Math.floor(getRand(0, childChromosome2.length));
            childChromosome2[mutationIndex] = getRand(-0.005, 0.005);
        }

        newGen.push(new MLP(nInputs, nHidden, nOutputs, childChromosome1));
        newGen.push(new MLP(nInputs, nHidden, nOutputs, childChromosome2));
    }

    return newGen;
}

function getSuccessRates(fitnesses)
{
    let sum = 0
    for(let i = 0; i < fitnesses.length; ++i)
    {
        sum += fitnesses[i];
    }

    let successRates = [];
    let prev = 0;
    for(let i = 0; i < fitnesses.length; ++i)
    {
        successRates.push(prev + fitnesses[i] / sum);
        prev = successRates[i];
    }

    return successRates;
}

function sortNeuralNets(neuralNets, fitnesses)
{
    for(let i = 0; i < fitnesses.length - 1; ++i)
    {
        if(fitnesses[i+ 1] < fitnesses[i])
        {
            for(let j = i; j >= 0 && fitnesses[j + 1] < fitnesses[j]; --j)
            {
                let temp = fitnesses[j];
                fitnesses[j] = fitnesses[j + 1];
                fitnesses[j + 1] = temp;

                temp = neuralNets[j];
                neuralNets[j] = neuralNets[j + 1];
                neuralNets[j + 1] = temp;
            }
        }
    }
}
