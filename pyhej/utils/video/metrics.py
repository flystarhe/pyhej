

if __name__ == "__main__":
    #http://nbviewer.jupyter.org/github/pyannote/pyannote-metrics/blob/master/notebooks/pyannote.metrics.diarization.ipynb
    %pylab inline

    from pyannote.core import Annotation, Segment
    reference = Annotation()
    reference[Segment(0, 10)] = "A"
    reference[Segment(12, 20)] = "B"
    reference[Segment(24, 27)] = "A"
    reference[Segment(30, 40)] = "C"
    reference

    hypothesis = Annotation()
    hypothesis[Segment(2, 13)] = "a"
    hypothesis[Segment(13, 14)] = "d"
    hypothesis[Segment(14, 20)] = "b"
    hypothesis[Segment(22, 38)] = "c"
    hypothesis[Segment(38, 40)] = "d"
    hypothesis

    #Diarization error rate
    from pyannote.metrics.diarization import DiarizationErrorRate
    diarizationErrorRate = DiarizationErrorRate()
    print("DER = {0:.3f}".format(diarizationErrorRate(reference, hypothesis, uem=Segment(0, 40))))
    #Optimal mapping
    diarizationErrorRate.optimal_mapping(reference, hypothesis)
    #Details
    diarizationErrorRate(reference, hypothesis, detailed=True, uem=Segment(0, 40))
    #Clusters purity and coverage
    from pyannote.metrics.diarization import DiarizationPurity
    purity = DiarizationPurity()
    print("Purity = {0:.3f}".format(purity(reference, hypothesis, uem=Segment(0, 40))))
    from pyannote.metrics.diarization import DiarizationCoverage
    coverage = DiarizationCoverage()
    print("Coverage = {0:.3f}".format(coverage(reference, hypothesis, uem=Segment(0, 40))))


    #http://nbviewer.jupyter.org/github/pyannote/pyannote-metrics/blob/master/notebooks/pyannote.metrics.identification.ipynb
    %pylab inline

    from pyannote.core import Annotation, Segment
    reference = Annotation()
    reference[Segment(0, 10)] = "A"
    reference[Segment(12, 20)] = "B"
    reference[Segment(24, 27)] = "A"
    reference[Segment(30, 40)] = "C"
    reference

    hypothesis = Annotation()
    hypothesis[Segment(2, 13)] = "A"
    hypothesis[Segment(13, 14)] = "D"
    hypothesis[Segment(14, 20)] = "B"
    hypothesis[Segment(22, 38)] = "C"
    hypothesis[Segment(38, 40)] = "D"
    hypothesis

    #Identification error rate
    from pyannote.metrics.identification import IdentificationErrorRate
    identificationErrorRate = IdentificationErrorRate()
    print("IER = {0:.3f}".format(identificationErrorRate(reference, hypothesis, uem=Segment(0, 40))))
    #Confusion matrix
    imshow(reference * hypothesis, interpolation="nearest"); colorbar();
    #Precision and coverage
    from pyannote.metrics.identification import IdentificationPrecision
    precision = IdentificationPrecision()
    print("Precision = {0:.3f}".format(precision(reference, hypothesis, uem=Segment(0, 40))))
    from pyannote.metrics.identification import IdentificationRecall
    recall = IdentificationRecall()
    print("Recall = {0:.3f}".format(recall(reference, hypothesis, uem=Segment(0, 40))))