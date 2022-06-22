import React from "react";
import BlurContainer from "../BlurContainer";
import PredictionEntry from "./PredictionEntry";
import "./Prediction.css";

type PredictionProps = {
    predictions?: { [label: string]: number },
    top?: number
}

function Prediction({predictions, top=1}: PredictionProps) {
    // display the top N (default 3) predictions returned from the model
    let sortedPredictions: Array<[string, number]> | undefined;
    if (!!predictions) {
        // sort our predictions by the confidence value and take the top N
        sortedPredictions = Object.entries(predictions)
            .sort((a, b) => b[1] - a[1])
            .slice(0, top);
    }
    return (
        <div id="predictions">
            <BlurContainer additionalClassname="prediction-container">
                {!!sortedPredictions ?
                    sortedPredictions.map(([label, confidence], idx) => (
                        <PredictionEntry key={label} label={label} confidence={Math.round(confidence*100 * 10) / 10} top={idx===0} />
                    ))
                    : <PredictionEntry label={'Loading...'} confidence={0} />
                }
            </BlurContainer>
        </div>
    );
}

export default Prediction;
