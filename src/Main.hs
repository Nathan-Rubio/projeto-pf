module Main where
import System.Random
import qualified Data.ByteString.Lazy as BL
import qualified Data.Csv as Csv
import Data.Vector (Vector)

-- ADT do Neurônio
data Neuronio = Neuronio {
    pesos    :: [Double],
    vies     :: Double,
    ativacao :: Double -> Double
}

-- Criação do tipo camada, que se trata de uma lista de neurônios
type Camada = [Neuronio]

-- Criação da ADT para uma Rede Neural, formada por camadas, 
-- que se é uma lista de elementos do tipo camada
newtype RedeNeural = RedeNeural {
    camadas :: [Camada]
}
 
-- Criação da ADT para representar a matriz de confusão
matrizConfusao :: [Double] -> [Double] -> [Resultado]
matrizConfusao previsoes reais = zipWith classificar
    where
        classificar previsao real
            | previsao >= 0.5 && real == 1.0 = TP
            | previsao >= 0.5 && real == 0.0 = FP
            | previsao <  0.5 && real == 0.0 = TN
            | otherwise                      = FN

-- Definição da função sigmoid
sigmoid :: Double -> Double
sigmoid z = 1 / (1 + exp (-z))

-- Definição da função relu
relu :: Double -> Double
relu = max 0

-- Função responsável por gerar o output do neurônio
neuronioOutput :: Neuronio -> [Double] -> Double
neuronioOutput (Neuronio ps v f) inputs =
    let z = sum (zipWith (*) ps inputs) + v
    in f z

-- Função responsável por gerar o output da camada, devolvendo o output de cada neurônio
camadaOutput :: Camada -> [Double] -> [Double]
camadaOutput camada inputs = map (\neuronio -> neuronioOutput neuronio inputs) camada

-- Função responsável por gerar o output da rede neural como um todo
redeNeuralOutput :: RedeNeural -> [Double] -> [Double]
redeNeuralOutput (RedeNeural []) inputs = inputs
redeNeuralOutput (RedeNeural (x:xs)) inputs =
    let output = camadaOutput x inputs
    in redeNeuralOutput (RedeNeural xs) output

-- Função para inicializar o peso de um neurônio qualquer de maneira randômica
inicializarPesos :: Int -> Int -> [Double]
inicializarPesos seed n = take n (randomRs (-1, 1) (mkStdGen seed))

-- Função para inicializar um peso randômico em todas as camadas da Rede Neural
inicializarRedeNeural :: [Int] -> Int -> RedeNeural
inicializarRedeNeural estrutura seed =
    let camadas = zipWith (\n_entrada n_saida -> [Neuronio (inicializarPesos seed n_entrada) 0 relu | _ <- [1..n_saida]]) estrutura (tail estrutura)
    in RedeNeural camadas

-- Função para o cálculo do Erro Quadrático Médio (Mean Squared Error - MSE)
mse :: [Double] -> [Double] -> Double
mse saida desejado = sum [erro ^ 2 | erro <- erros] / fromIntegral (length saida)
    where
        erros = [s - d | (s, d) <- zip saida desejado]

-- Função para o cálculo do gradiente simplificado
gradiente :: Double -> Double
gradiente saida = saida * 0.01

atualizarNeuronio :: Neuronio -> [Double] -> [Double] -> Neuronio
atualizarNeuronio (Neuronio pesos vies ativacao) inputs gradientes =
    Neuronio novosPesos vies ativacao
    where
        -- 0.1 - Taxa de Aprendizagem
        novosPesos = zipWith(\peso grad -> peso - 0.1 * grad) pesos gradientes

treinarRedeNeural :: RedeNeural -> [([Double], [Double])] -> Int -> RedeNeural
treinarRedeNeural rede dataset epocas = foldl treinarEpoca rede [1..epocas]
    where
        treinarEpoca redeAtual _ = foldl (\rede (inputs, desejado) ->
            let saida      = redeNeuralOutput redeAtual inputs
                erro       = mse saida desejado
                gradientes = map gradiente saida
                novaRede   = atualizarPesos redeAtual inputs gradientes
            in novaRede) redeAtual dataset

-- Função para atualizar a rede com novos pesos
atualizarPesos :: RedeNeural -> [Double] -> [Double] -> RedeNeural
atualizarPesos (RedeNeural camadas) inputs gradientes =
    RedeNeural (map (\camada -> map (\neuronio -> atualizarNeuronio neuronio inputs gradientes) camada) camadas)

redeExemplo :: RedeNeural
redeExemplo = RedeNeural [
    [Neuronio [0.5, -0.2] 0.1 relu, Neuronio [0.3, 0.8] (-0.4) relu], --Primeira Camada
    [Neuronio [0.7,  0.3] 0.2 sigmoid]                                -- Camada de Saída
    ]


main :: IO ()
main = do
    let rede         = inicializarRedeNeural [2, 2, 1] 42
    let dataset      = [([0.5, -0.3], [1.0]), ([0.1, 0.8], [0.0])]
    let redeTreinada = treinarRedeNeural rede dataset 1000
    let resultado    = redeNeuralOutput redeTreinada [0.9, -0.1]

    putStrLn $ "Resultado da Rede Neural Treinada: " ++ show resultado