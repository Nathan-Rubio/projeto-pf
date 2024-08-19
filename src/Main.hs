module Main where
import System.Random

-- Criação de Tipos para melhor visualização
type Pesos    = [Double]
type Vies     = Double
type Ativacao = Double -> Double -- Função de ativação do neurônio (Neste caso, RELU ou Sigmoid)

-- ADT do Neurônio
data Neuronio = Neuronio {
    pesos    :: Pesos,
    vies     :: Vies,
    ativacao :: Ativacao
}

-- Criação do tipo camada, que se trata de uma lista de neurônios
type Camada = [Neuronio]

-- Criação da ADT para uma Rede Neural, formada por camadas
newtype RedeNeural = RedeNeural {
    camadas :: [Camada]
}

-- ADT do Resultado da Matriz de Confusão (Ainda não Implementada)
data Resultado = TP | FP | TN | FN -- Determina os 4 valores da matriz de confusão
    deriving (Show, Eq)

-- Função para a Matriz de Confusão (Ainda não Implementada)
matrizConfusao :: [Double] -> [Double] -> [Resultado]
matrizConfusao previsoes reais = zipWith classificar previsoes reais
    where
        classificar previsao real
            | previsao >= 0.5 && real == 1.0 = TP  -- Previu positivo e o real era positivo (correto)
            | previsao >= 0.5 && real == 0.0 = FP  -- Previu positivo, mas o real era negativo (erro)
            | previsao < 0.5  && real == 0.0 = TN  -- Previu negativo e o real era negativo (correto)
            | otherwise                      = FN  -- Previu negativo, mas o real era positivo (erro)

-- Definição da função relu
relu :: Double -> Double
relu = max 0

-- Definição da função sigmoid
sigmoid :: Double -> Double
sigmoid z = 1 / (1 + exp (-z))

type Inputs = [Double]
type Output = Double

-- Função responsável por gerar o output do neurônio
-- Multiplica-se o peso com cada input, somando todos e adicionando o viés, obtendo z
-- z então será utilizado na função de ativação f
neuronioOutput :: Neuronio -> Inputs -> Output
neuronioOutput (Neuronio ps v f) inputs =
    let z = sum (zipWith (*) ps inputs) + v
    in f z

-- Função responsável por gerar o output da camada, devolvendo o output de cada neurônio
camadaOutput :: Camada -> Inputs -> [Output]
camadaOutput camada inputs = map (\neuronio -> neuronioOutput neuronio inputs) camada

-- Função responsável por gerar o output da rede neural como um todo
redeNeuralOutput :: RedeNeural -> Inputs -> [Output]
redeNeuralOutput (RedeNeural []) inputs = inputs -- Rede Vazia devolve os inputs
redeNeuralOutput (RedeNeural (c:cs)) inputs =    -- Passa por cada camada recursivamente, usando o output da anterior
    let output = camadaOutput c inputs
    in redeNeuralOutput (RedeNeural cs) output

type Seed = Int

-- Função para inicializar os pesos de um neurônio de maneira randômica
inicializarPesos :: Seed -> Int -> Pesos
inicializarPesos seed n = take n (randomRs (-1, 1) (mkStdGen seed))

-- Função para inicializar os pesos randômicos em todas as camadas da Rede Neural
inicializarRedeNeural :: [Int] -> Seed -> RedeNeural
inicializarRedeNeural estrutura seed =
    -- Para cada camada inicializa a quantidade de neurônios recebida, 
    -- com o peso randômico e o viés em 0, usando a função relu
    let cs = zipWith (\n_entrada n_saida -> [Neuronio (inicializarPesos seed n_entrada) 0 relu | _ <- [1..n_saida]]) (init estrutura) (tail estrutura)
        -- Para a última camada, inicializa apenas um neurônio que utiliza a função sigmoid
        ultimaCamada = [Neuronio (inicializarPesos seed (last estrutura)) 0 sigmoid]
    in RedeNeural (cs ++ [ultimaCamada])

-- Função para o cálculo do Erro Quadrático Médio (Mean Squared Error - MSE)
mse :: [Double] -> [Double] -> Double
mse saida desejado = sum [erro ** 2 | erro <- erros] / fromIntegral (length saida)
    where
        erros = [s - d | (s, d) <- zip saida desejado]

-- Função para o cálculo do gradiente simplificado
gradienteSigmoid :: Double -> Double
gradienteSigmoid saida = saida * 0.01

-- Função para atualizar a rede com novos pesos
atualizarPesos :: RedeNeural -> Inputs -> [Double] -> RedeNeural
atualizarPesos (RedeNeural cs) inputs gradientes =
    -- Atualiza cada neurônio em uma camada e mapeia, depois mapeia essa camada para as camadas da Rede Neural
    RedeNeural (map (\c -> map (\neuronio -> atualizarNeuronio neuronio inputs gradientes) c) cs)

-- Função para atualizar os pesos do Neurônio
atualizarNeuronio :: Neuronio -> Inputs -> [Double] -> Neuronio
atualizarNeuronio (Neuronio ps v f) inputs gradientes =
    let taxaAprendizado = 0.1 -- Ajusta a velocidade de aprendizado
        novosPesos = zipWith (\p (input, grad) -> p - taxaAprendizado * grad * input) ps (zip inputs gradientes)
        novoVies   = v - taxaAprendizado * sum gradientes
    in Neuronio novosPesos novoVies f
        
-- Criação de tipos para facilitar a leitura
type Dataset = [(Inputs, Inputs)] -- Dataset com dois tipos de inputs
type Epocas  = Int

-- Função para o treinamento da Rede Neural
treinarRedeNeural :: RedeNeural -> Dataset -> Epocas -> IO RedeNeural
treinarRedeNeural rede dataset epocas = treinar rede epocas
    where
        treinar redeAtual 0 = return redeAtual -- Se não tiver mais épocas, retorna a rede atual
        treinar redeAtual n = do
            -- Para cada exemplo no dataset, faça:
            let novaRede = foldl (\rede (inputs, desejado) ->
                    let saida      = redeNeuralOutput rede inputs          -- Calcula a saída da rede para a entrada atual
                        erro       = mse saida desejado                    -- Calcula o erro entre a saída prevista e a desejada
                        gradientes = map gradienteSigmoid saida            -- Calcula os gradientes da função de ativação na saída
                        novaRede   = atualizarPesos rede inputs gradientes -- Atualiza os pesos da rede com base no erro e nos gradientes
                    in novaRede) redeAtual dataset
                -- Monitorando o erro e pesos após cada época
                erroTotal = sum [mse (redeNeuralOutput novaRede inputs) desejado | (inputs, desejado) <- dataset] / fromIntegral (length dataset)
            putStrLn $ "Época: " ++ show (epocas - n + 1) ++ ", Erro Total: " ++ show erroTotal
            mapM_ printCamada (camadas novaRede)
            putStrLn ""
            treinar novaRede (n - 1) -- Chama a função recursivamente para a próxima época

-- Função para imprimir os pesos de uma camada
printCamada :: Camada -> IO ()
printCamada camada = do
    mapM_ printNeuronio camada
    putStrLn ""

-- Função para imprimir os pesos de um neurônio
printNeuronio :: Neuronio -> IO ()
printNeuronio (Neuronio ps v _) = putStrLn $ "Pesos: " ++ show ps ++ ", Vies: " ++ show v

main :: IO ()
main = do
    -- Conjunto de dados de treinamento normalizados
    let datasetTreinamento = [([0.444, 0.500], [0.0]), 
                              ([0.556, 0.625], [1.0]), 
                              ([0.333, 0.125], [0.0]),
                              ([0.878, 0.825], [1.0]),
                              ([0.100, 0.228], [0.0]),
                              ([0.738, 0.611], [1.0]),
                              ([0.456, 0.222], [0.0]),
                              ([0.667, 0.773], [1.0])
                             ]

    -- Inicia a rede neural com 3 camadas
    -- 2 neurônios - 1° camada
    -- 2 neurônios - 2° camada
    -- 1 neurônio  - 3° camada (saída)
    -- 42 - Semente usada para esta Rede
    let rede = inicializarRedeNeural [2, 2, 1] 42
    putStrLn "Pesos Iniciais: "
    mapM_ printCamada (camadas rede)

    -- Obtém-se a rede neural treinada
    -- Usa-se o datasetTreinamento em 10 épocas
    redeTreinada <- treinarRedeNeural rede datasetTreinamento 10
    
    -- Resultados de testes básicos
    let resultado1 = redeNeuralOutput redeTreinada [0.945, 0.832]
    let resultado2 = redeNeuralOutput redeTreinada [0.211, 0.102]
    let resultado3 = redeNeuralOutput redeTreinada [0.811, 0.418]

    putStrLn $ "Resultado 1: " ++ show resultado1
    putStrLn $ "Resultado 2: " ++ show resultado2
    putStrLn $ "Resultado 3: " ++ show resultado3
