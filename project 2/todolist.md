## Kode

# Part A:
Har unnagjort første 2 deloppgaver - Plain Gradient Descent(PGD) og Momentum PGD.
Startet på kode for SGD. Mangler å tenke ut hvordan man skal bestemme
kriterium for å stoppe beregning.
Virker som man starter en helt ny beregning av gradient i hver minibatch, men gir dette mening?
Eller burde man gjenbruke betaene fra foregående minibatches - men evaluere de på de nye batchene når man minimerer kostfunksjonen?
 _Må leses på!_
 
**Ting jeg lurer på**
- Når vi kjører SGD med minibatches, velger vi ofte flere epochs enn vi har batches. Dette blir jo større numerisk belastning enn å bare ta gradienten over hele datasettet? (Flere iterasjoner). Er det normalt at vi har færre epochs enn minibatches egentlig? **NEI**


## NN
Man kjører "bakover" gjennom nettverket, og gjør GD når man har funnet et uttrykk
