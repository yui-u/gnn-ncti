import bisect
import copy
import math
from pathlib import Path

import music21 as m21
import numpy as np
import torch
import torch.nn.functional as F
from music21.chord import Chord
from music21.corpus.chorales import ChoraleListRKBWV
from music21.interval import Interval
from music21.note import Note, Rest
from music21.pitch import Pitch
from music21.roman import RomanNumeral
from music21.stream import Measure, Voice
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph

from core.common.constants import *
from core.preprocess.instances import ListInstance, ValueInstance

TIME_NODE_TYPE_LABEL = 0
NOTE_NODE_TYPE_LABEL = 1
DUMMY_NOTE_FEAT = -1
DUMMY_EDGE_FEAT = -1

NODE_FEATS = {
    'offset': 0,
    'beat': 1,
    'length': 2,
    'midi': 3,
    'pc': 4,
}

NODE_LABELS = {
    'part': 5,
    'voice': 6,
    'node_type': 7
}

EDGE_LABEL = {
    'neighbor': 0,
    'overlap': 1,
    'self_note': 2,
    't2t': 3,
    't2n': 4,
    'self_time': 5
}

NCT_CHORD_INDEX = 0
NCT_NONCHORD_INDEX = 1
NCT_BOTH_INDEX = 2
NCT_IGNORE_INDEX = 3

BACH_CHORALE_DUPLICATE_LIST = [
    [5, 309],
    [9, 361],
    [23, 88],
    [53, 178],
    [64, 256],
    [86, 195, 305],
    [91, 259],
    [93, 257],
    [100, 126],
    [120, 349],
    [125, 326],
    [131, 328],
    [144, 318],
    [156, 308],
    [198, 307],
    [199, 302],
    [201, 306],
    [235, 319],
    [236, 295],
    [248, 354],
    [254, 282],
    [313, 353],
]

BACH_CHORALE_INVALID_MEASURE_LIST = [
    130,  # It is in 4/4, but the eighth measure is only 2 x quarter long.
]

LITTLE_ORGAN_UNINTERPRETABLE_BY_TONALITY = {
    'BWV614',
    'BWV621',
}


class MxlReader(object):
    def __init__(
            self,
            dataset,
            cv_num_set,
            resolution_str,
            chord_type,
            alert_nct_ratio_th,
            dir_output_dataset
    ):
        self._small_eps = 1e-13
        self._large_eps = 1e-5
        self._dataset = dataset
        self._cv_num_set = cv_num_set
        self._resolution_str = resolution_str
        self._chord_type = chord_type
        self._alert_nct_ratio_th = alert_nct_ratio_th
        self._dir_output_dataset = dir_output_dataset

    @staticmethod
    def _check_key_signatures(score):
        keysig_sharps = []
        for pi in range(len(score.parts)):
            for mi, m in enumerate(score.parts[pi].getElementsByClass(Measure)):
                if hasattr(m, 'keySignature') and m.keySignature is not None:
                    if m.keySignature.sharps not in keysig_sharps:
                        keysig_sharps.append(m.keySignature.sharps)
        valid = 1 == len(keysig_sharps)
        return keysig_sharps[0], valid

    @staticmethod
    def _check_measure_alignment(score):
        ms = [(m.number, m.offset) for m in score.parts[0].getElementsByClass(Measure)]
        for pi in range(1, len(score.parts)):
            if ms != [(m.number, m.offset) for m in score.parts[pi].getElementsByClass(Measure)]:
                print('Inconsistent measures')
                return False
        return True

    @staticmethod
    def _check_measure_offset_consistency(logger, offset2mn, rntxt, end_margin=2):
        score_measures = sorted(list(set([v['m_number'] for v in offset2mn.values()])))
        rntxt_measures = sorted(list(set([v.measureNumber for v in rntxt.flat.elements if v.measureNumber is not None])))
        if score_measures != rntxt_measures:
            if score_measures[0] == rntxt_measures[0]:
                if 0 <= (score_measures[-1] - rntxt_measures[-1]) <= end_margin:
                    pass  # ok. rntxt may end a few bars early.
                else:
                    logger.info('Inconsistent set of measures: {}, {}'.format(score_measures[-1], rntxt_measures[-1]))
                    return False
            else:
                logger.info('Inconsistent set of measures: ({}, {}), ({}, {})'.format(
                    score_measures[0], score_measures[-1], rntxt_measures[0], rntxt_measures[-1]))
                return False

        # first item
        first_score_offset = list(offset2mn.keys())[0]
        first_elem_rntxt = None
        first_measure_rntxt = rntxt.parts[0].getElementsByClass(Measure)[0]
        for e in first_measure_rntxt.elements:
            if isinstance(e, RomanNumeral):
                first_elem_rntxt = e
                break

        # auftakt
        if 0 == score_measures[0]:
            # auftakt (score)
            offset = 0.0
            for offset in offset2mn:
                if 0 < offset2mn[offset]['m_number']:
                    break
            assert 1 == offset2mn[offset]['m_number']
            assert abs(offset2mn[offset]['beat'] - 1.0) < 1e-7
            score_auftakt_offset = offset - first_score_offset
            # auftakt (rntxt)
            rntxt_auftakt_offset = None
            for m in rntxt.parts[0].getElementsByClass(Measure):
                if 0 < m.measureNumber:
                    for e in m.elements:
                        if isinstance(e, RomanNumeral):
                            assert abs(e.beat - 1.0) < 1e-7, e.beat
                            rntxt_auftakt_offset = (m.offset + e.offset) - first_elem_rntxt.offset
                            break
                    break
            if score_auftakt_offset != rntxt_auftakt_offset:
                logger.info('Inconsistent first auftakt offsets between score and analysis')
                return False

        # offset consistency
        if offset2mn[first_score_offset]['m_number'] != first_measure_rntxt.measureNumber:
            logger.info('Inconsistent first measure between score and analysis')
            if offset2mn[first_score_offset]['n_offset'] != first_elem_rntxt.offset:
                logger.info('Inconsistent first offset between score and analysis: {}, {}'.format(
                    offset2mn[first_score_offset]['n_offset'], first_elem_rntxt.offset))
            return False
        else:
            if offset2mn[first_score_offset]['n_offset'] != first_elem_rntxt.offset:
                logger.info('Inconsistent first offset between score and analysis: {}, {}'.format(
                    offset2mn[first_score_offset]['n_offset'], first_elem_rntxt.offset))
                return False
            else:
                return True

    def _get_resolution(self, score):
        timesig = score.parts[0].measure(2).getContextByClass('TimeSignature')
        # resolution
        if self._resolution_str == '8th':
            resolution = 0.5
        elif self._resolution_str == '16th':
            resolution = 0.25
        elif self._resolution_str == 'halfbeat':
            if '{}/{}'.format(timesig.numerator, timesig.denominator) in ['2/2', '3/2', '2/4', '3/4', '4/4', '6/4']:
                resolution = (4.0 / float(timesig.denominator)) * 0.5
            elif '{}/{}'.format(timesig.numerator, timesig.denominator) in ['3/8', '4/8', '6/8', '9/8', '12/8']:
                resolution = 0.5
            elif '{}/{}'.format(timesig.numerator, timesig.denominator) in ['4/16', '12/16']:
                resolution = 0.25
            else:
                print(timesig)
                raise NotImplementedError
        elif self._resolution_str == 'beat':
            if '{}/{}'.format(timesig.numerator, timesig.denominator) in ['2/2', '3/2', '2/4', '3/4', '4/4', '4/6']:
                resolution = 4.0 / float(timesig.denominator)
            elif '{}/{}'.format(timesig.numerator, timesig.denominator) in ['3/8', '4/8', '6/8', '9/8' '12/8']:
                resolution = 1.5
            elif '{}/{}'.format(timesig.numerator, timesig.denominator) in ['4/16', '12/16']:
                resolution = 0.75
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        # time per measure
        if '{}/{}'.format(timesig.numerator, timesig.denominator) in ['2/2', '3/2', '2/4', '3/4', '4/4', '6/4']:
            time_per_measure = (4.0 * timesig.numerator) / float(timesig.denominator)
        elif '{}/{}'.format(timesig.numerator, timesig.denominator) in ['3/8', '4/8', '6/8', '9/8', '12/8']:
            time_per_measure = 0.5 * timesig.numerator
        elif '{}/{}'.format(timesig.numerator, timesig.denominator) in ['4/16', '12/16']:
            time_per_measure = 0.25 * timesig.numerator
        else:
            raise NotImplementedError

        return timesig, resolution, time_per_measure

    @staticmethod
    def _get_notes(score):
        notes = {}
        last_note_offset = 0.0
        for pi, part in enumerate(score.parts):
            for m in part.getElementsByClass(Measure):
                m_iter = [v for v in m.getElementsByClass(Voice)]
                if bool(m.elements):
                    m_iter += [m]
                for vi, v in enumerate(m_iter):
                    if (pi, vi) not in notes:
                        notes[(pi, vi)] = []
                    for n in [vn for vn in v.elements if isinstance(vn, Note) or isinstance(vn, Rest) or isinstance(vn, Chord)]:
                        if 0.0 < n.quarterLength:  # ignore invalid notes
                            notes[(pi, vi)].append((m, n))
                            if last_note_offset < m.offset + n.offset + n.quarterLength:
                                last_note_offset = m.offset + n.offset + n.quarterLength

        # sort key
        notes = dict(sorted(notes.items()))
        # sort value and calc. coverage
        for pk, pv in notes.items():
            pvo = []
            for v in pv:
                n_offset = v[1].offset
                pvo.append((v[0].offset + n_offset, v))
            pvo = sorted(pvo, key=lambda x:x[0])
            notes[pk] = [v[1] for v in pvo]
        return notes, last_note_offset

    def _get_score_as_graph(self, score):
        timesig, resolution, time_per_measure = self._get_resolution(score)
        notes, last_note_offsets = self._get_notes(score)
        timesteps = [float(i * resolution) for i in range(int(math.ceil(last_note_offsets / resolution)))]
        voice_coverage = {}
        for pk, pv in notes.items():
            voice_coverage[pk] = sum(v[1].quarterLength for v in pv) / last_note_offsets

        # note nodes and neighbor-notes edges
        note_feature = [[] for _ in range(len(NODE_FEATS) + len(NODE_LABELS))]
        neighbor_note_edges = [[], []]
        neighbor_note_edge_feat = []
        self_note_edges = [[], []]
        self_note_edge_feat = []
        raw_ann = {}
        note_id_start = len(timesteps)
        note_id = note_id_start
        offset2mn = {}
        fermata_positions = []
        for pk, part_mn in notes.items():
            # new part or voice
            prev_note_ids = []
            for pmn in part_mn:
                cur_note_ids = []
                m, pn = pmn
                if isinstance(pn, Chord):
                    pns = pn.notes
                else:
                    pns = [pn]
                # remove grace notes
                pns = [_ for _ in pns if not _.duration.isGrace]
                # Some notes in the chord object do not have the correct offset.
                n_offset = pn.offset
                for n in pns:
                    # timestep to measure, note offsets
                    if m.offset + n_offset in offset2mn:
                        assert (
                            (round(offset2mn[m.offset + n_offset]['m_number'], 0) == round(m.number, 0)) and
                            (round(offset2mn[m.offset + n_offset]['m_paddingLeft'], 4) == round(m.paddingLeft, 4)) and
                            (round(offset2mn[m.offset + n_offset]['m_offset'], 4) == round(m.offset, 4)) and
                            (round(offset2mn[m.offset + n_offset]['n_offset'], 4) == round(n_offset, 4))
                        )
                        if math.isnan(offset2mn[m.offset + n_offset]['beat']) and not math.isnan(n.beat):
                            offset2mn[m.offset + n_offset]['beat'] = n.beat
                    else:
                        offset2mn[m.offset + n_offset] = {
                            'm_number': m.number, 'm_paddingLeft': m.paddingLeft, 'm_offset': m.offset,
                            'n_offset': n_offset, 'beat': n.beat}
                    # fermata
                    for e in n.expressions:
                        if e.name == 'fermata':
                            if m.offset + n_offset not in fermata_positions:
                                fermata_positions.append(m.offset + n_offset + n.quarterLength)
                    # neighbor edges
                    for prev_i in prev_note_ids:
                        neighbor_note_edges[0].append(prev_i)
                        neighbor_note_edges[1].append(note_id)
                        neighbor_onset_diff = float(m.offset + n_offset) - note_feature[NODE_FEATS['offset']][prev_i - note_id_start]
                        neighbor_note_edge_feat.append([EDGE_LABEL['neighbor'], neighbor_onset_diff])
                        # reverse
                        neighbor_note_edges[0].append(note_id)
                        neighbor_note_edges[1].append(prev_i)
                        neighbor_note_edge_feat.append([EDGE_LABEL['neighbor'], -neighbor_onset_diff])
                    # self note edge
                    self_note_edges[0].append(note_id)
                    self_note_edges[1].append(note_id)
                    self_note_edge_feat.append([EDGE_LABEL['self_note'], float(n.quarterLength)])
                    # note features
                    note_feature[NODE_LABELS['node_type']].append(NOTE_NODE_TYPE_LABEL)
                    note_feature[NODE_LABELS['part']].append(pk[0])
                    note_feature[NODE_LABELS['voice']].append(pk[1])
                    note_feature[NODE_FEATS['offset']].append(float(m.offset + n_offset))
                    note_feature[NODE_FEATS['beat']].append(float(n.beat) if not math.isnan(n.beat) else 0.0)
                    note_feature[NODE_FEATS['length']].append(float(n.quarterLength))
                    if isinstance(n, Rest):
                        note_feature[NODE_FEATS['midi']].append(REST_INDEX)
                        note_feature[NODE_FEATS['pc']].append(REST_INDEX)
                    else:
                        note_feature[NODE_FEATS['midi']].append(n.pitch.midi)
                        note_feature[NODE_FEATS['pc']].append(n.pitch.midi % 12)
                    cur_note_ids.append(note_id)
                    note_id += 1
                # next edge
                prev_note_ids = cur_note_ids.copy()

        # sort offset2mn
        offset2mn = dict(sorted(offset2mn.items(), key=lambda x: x[0]))

        # sort raw ann
        raw_ann = dict(sorted(raw_ann.items(), key=lambda x: x[0]))

        # node
        assert note_id == len(note_feature[0]) + len(timesteps)
        data = Data()
        note_feature = torch.tensor(note_feature).transpose(0, 1)  # (N_notes, NOTE_FEAT)
        time_feature = torch.ones(len(timesteps), len(NODE_FEATS) + len(NODE_LABELS)) * DUMMY_NOTE_FEAT
        time_feature[:, NODE_FEATS['offset']] = torch.tensor(timesteps)
        time_feature[:, NODE_FEATS['length']] = torch.ones(len(timesteps)) * resolution
        time_feature[:, NODE_LABELS['node_type']] = torch.ones(len(timesteps)) * TIME_NODE_TYPE_LABEL
        x = torch.cat([time_feature, note_feature], dim=0)
        data.x = x

        # note overlap edges
        overlap_note_edges = [[], []]
        overlap_note_edge_feat = []
        for note_i in range(len(timesteps), data.x.size(0) - 1):
            ni_start = data.x[note_i, NODE_FEATS['offset']].item()
            ni_end = ni_start + data.x[note_i, NODE_FEATS['length']].item()
            for note_j in range(note_i + 1, data.x.size(0)):
                nj_start = data.x[note_j, NODE_FEATS['offset']].item()
                nj_end = nj_start + data.x[note_j, NODE_FEATS['length']].item()
                overlap = min(ni_end, nj_end) - max(ni_start, nj_start)
                if 0.0 < overlap:
                    overlap_note_edges[0].append(note_i)
                    overlap_note_edges[1].append(note_j)
                    overlap_note_edge_feat.append([EDGE_LABEL['overlap'], overlap])
                    # reverse
                    overlap_note_edges[0].append(note_j)
                    overlap_note_edges[1].append(note_i)
                    overlap_note_edge_feat.append([EDGE_LABEL['overlap'], overlap])

        # time to time edges, time to note edges
        time2time_edges = [[], []]
        time2time_edge_feat = []
        time2note_edges = [[], []]
        time2note_edge_feat = []
        self_time_edges = [[], []]
        self_time_edge_feat = []
        for t in range(len(timesteps)):
            t_start = t * resolution
            t_end = (t + 1) * resolution
            self_time_edges[0].append(t)
            self_time_edges[1].append(t)
            self_time_edge_feat.append([EDGE_LABEL['self_time'], resolution])
            if t + 1 < len(timesteps):
                time2time_edges[0].append(t)
                time2time_edges[1].append(t + 1)
                time2time_edge_feat.append([EDGE_LABEL['t2t'], resolution])
            for note_id in range(len(timesteps), data.x.size(0)):
                n_start = data.x[note_id, NODE_FEATS['offset']].item()
                n_end = n_start + data.x[note_id, NODE_FEATS['length']].item()
                # range of overlap
                overlap = min(t_end, n_end) - max(t_start, n_start)
                if 0.0 < overlap:
                    time2note_edges[0].append(t)
                    time2note_edges[1].append(note_id)
                    time2note_edge_feat.append([EDGE_LABEL['t2n'], overlap])

        edge_index = [None] * len(EDGE_LABEL)
        edge_index[EDGE_LABEL['neighbor']] = torch.tensor(neighbor_note_edges)
        edge_index[EDGE_LABEL['overlap']] = torch.tensor(overlap_note_edges)
        edge_index[EDGE_LABEL['self_note']] = torch.tensor(self_note_edges)
        edge_index[EDGE_LABEL['t2t']] = torch.tensor(time2time_edges)
        edge_index[EDGE_LABEL['t2n']] = torch.tensor(time2note_edges)
        edge_index[EDGE_LABEL['self_time']] = torch.tensor(self_time_edges)
        data.edge_index = torch.cat(edge_index, dim=1)

        edge_attr = [None] * len(EDGE_LABEL)
        edge_attr[EDGE_LABEL['neighbor']] = torch.tensor(neighbor_note_edge_feat)
        edge_attr[EDGE_LABEL['overlap']] = torch.tensor(overlap_note_edge_feat)
        edge_attr[EDGE_LABEL['self_note']] = torch.tensor(self_note_edge_feat)
        edge_attr[EDGE_LABEL['t2t']] = torch.tensor(time2time_edge_feat)
        edge_attr[EDGE_LABEL['t2n']] = torch.tensor(time2note_edge_feat)
        edge_attr[EDGE_LABEL['self_time']] = torch.tensor(self_time_edge_feat)
        data.edge_attr = torch.cat(edge_attr, dim=0)

        return data, timesig, resolution, time_per_measure, voice_coverage, offset2mn, raw_ann, fermata_positions

    def _get_nct_annotation(self, graph, ann, logger):
        offset2nid = {}
        for nid in range(graph.x.size(0)):
            nid_offset = graph.x[nid, NODE_FEATS['offset']].item()
            if nid_offset not in offset2nid:
                offset2nid[nid_offset] = []
            offset2nid[nid_offset].append(nid)
        offset2nid = sorted(offset2nid.items(), key=lambda x:x[0])

        gold = list(ann.values())
        assert gold[0]['offset'] <= 0.0
        nct_ann = torch.ones((graph.x.size(0), 1)).long() * NCT_IGNORE_INDEX
        gi = 0
        for t, nids in offset2nid:
            while (gi < len(gold) - 1) and (gold[gi + 1]['offset'] <= t):
                gi += 1
            rns = [gold[gi]['rn']]
            if gold[gi]['rn'].pivotChord is not None:
                if gold[gi]['rn'].orderedPitchClasses != gold[gi]['rn'].pivotChord.orderedPitchClasses:
                    logger.info('Inconsistent pivot chord: (m={}), {}, {}'.format(
                        gold[gi]['measure'],
                        gold[gi]['rn'].pitchNames,
                        gold[gi]['rn'].pivotChord.pitchNames
                    ))
                rns.append(gold[gi]['rn'].pivotChord)
            chord_pcs = []
            for rn in rns:
                if 'triad' in self._chord_type:
                    chord_pcs.append(rn.root().pitchClass)
                    if rn.third is not None:
                        chord_pcs.append(rn.third.pitchClass)
                    if rn.fifth is not None:
                        chord_pcs.append(rn.fifth.pitchClass)
                    if 'dominant' in self._chord_type:
                        if 5 == rn.scaleDegree and 'major' in rn.quality.lower():
                            if 'little-organ' in self._dataset:
                                chord_pcs.append((rn.fifth.pitchClass + 3) % 12)
                            else:
                                if rn.seventh is not None:
                                    chord_pcs.append(rn.seventh.pitchClass)
                else:
                    assert 'full' == self._chord_type
                    chord_pcs += rn.orderedPitchClasses
            chord_pcs = sorted(list(set(chord_pcs)))
            for ni in nids:
                ni_pc = graph.x[ni, NODE_FEATS['pc']].long().item()
                if ni_pc != REST_INDEX:  # not REST
                    nct_label = NCT_CHORD_INDEX if (ni_pc in chord_pcs) else NCT_NONCHORD_INDEX
                    if nct_ann[ni, -1] == NCT_IGNORE_INDEX:
                        nct_ann[ni, -1] = nct_label
                    else:
                        if nct_ann[ni, -1] != nct_label:
                            nct_ann[ni, -1] = NCT_BOTH_INDEX
                        else:
                            nct_ann[ni, -1] = nct_label
        return nct_ann

    @staticmethod
    def generate_nct_score(org_score, graph_x, nct_gold, nct_pred):
        color_TP = 'lightsalmon'
        color_FP = 'forestgreen'
        color_FN = 'paleturquoise'
        nct_score = copy.deepcopy(org_score)
        pvo2nid = {}
        for nid in range(graph_x.size(0)):
            pi = graph_x[nid, NODE_LABELS['part']].item()
            vi = graph_x[nid, NODE_LABELS['voice']].item()
            offset = round(graph_x[nid, NODE_FEATS['offset']].item(), 3)
            if (graph_x[nid, NODE_LABELS['node_type']] == NOTE_NODE_TYPE_LABEL) and (0 <= graph_x[nid, NODE_FEATS['pc']]):  # not Rest
                if (pi, vi, offset) in pvo2nid:
                    # the case of Chord
                    pvo2nid[(pi, vi, offset)].append(nid)
                else:
                    pvo2nid[(pi, vi, offset)] = [nid]

        seen_nids = []
        for pi, part in enumerate(nct_score.parts):
            for m in part.getElementsByClass(Measure):
                m_iter = [v for v in m.getElementsByClass(Voice)]
                if bool(m.elements):
                    m_iter += [m]
                for vi, v in enumerate(m_iter):
                    for e in [vn for vn in v.elements if (isinstance(vn, Note) or isinstance(vn, Chord)) and (not vn.duration.isGrace)]:
                        offset = round(m.offset + e.offset, 3)
                        nids = pvo2nid[(pi, vi, offset)]
                        if isinstance(e, Chord):
                            midi2nctgold = {}
                            midi2nctpred = {}
                            for nid in nids:
                                midi = graph_x[nid, NODE_FEATS['midi']].long().item()
                                gold_nctlabel = nct_gold[nid, 0].long().item()
                                pred_nctlabel = nct_pred[nid, 0].long().item()
                                if midi in midi2nctgold:
                                    assert midi2nctgold[midi] == gold_nctlabel
                                    assert midi2nctpred[midi] == pred_nctlabel
                                else:
                                    midi2nctgold[midi] = gold_nctlabel
                                    midi2nctpred[midi] = pred_nctlabel
                                seen_nids.append(nid)
                            for note in e.notes:
                                assert note.pitch.midi in midi2nctgold
                                assert note.pitch.midi in midi2nctpred
                                if midi2nctgold[note.pitch.midi] in [NCT_NONCHORD_INDEX, NCT_BOTH_INDEX]:
                                    if midi2nctpred[note.pitch.midi] in [NCT_NONCHORD_INDEX, NCT_BOTH_INDEX]:
                                        note.style.color = color_TP
                                    else:
                                        note.style.color = color_FN
                                else:
                                    if midi2nctpred[note.pitch.midi] in [NCT_NONCHORD_INDEX, NCT_BOTH_INDEX]:
                                        note.style.color = color_FP
                                    else:
                                        # TN
                                        pass
                        else:
                            assert 1 == len(nids)
                            nid = nids[0]
                            if nct_gold[nid, 0] in [NCT_NONCHORD_INDEX, NCT_BOTH_INDEX]:
                                if nct_pred[nid, 0] in [NCT_NONCHORD_INDEX, NCT_BOTH_INDEX]:
                                    e.style.color = color_TP
                                else:
                                    e.style.color = color_FN
                            else:
                                if nct_pred[nid, 0] in [NCT_NONCHORD_INDEX, NCT_BOTH_INDEX]:
                                    e.style.color = color_FP
                                else:
                                    # TN
                                    pass
                            seen_nids.append(nid)

        assert len(seen_nids) == torch.where(0.0 <= graph_x[:, NODE_FEATS['pc']])[0].size(0)
        return nct_score

    def create_instance(self, logger):
        logger.info('Create instance')
        dir_mxl = Path(self._dataset)
        if 'Chorales' in self._dataset:
            duplicated_list = []
            for dpl in BACH_CHORALE_DUPLICATE_LIST:
                for d in dpl[1:]:  # choose smaller number
                    duplicated_list.append(d)
            mxl_files = []
            for riemen, v in ChoraleListRKBWV().byRiemenschneider.items():
                if riemen not in duplicated_list:
                    mxl_files.append((riemen, 'bwv{}'.format(v['bwv'])))
        else:
            mxl_files = sorted(dir_mxl.glob('*.musicxml')) + sorted(dir_mxl.glob('*.mxl')) + sorted(
                dir_mxl.glob('*.xml')) + sorted(dir_mxl.glob('*.krn'))
        split_dict = dict([(i, []) for i in range(self._cv_num_set)])

        instances_key_preprocess_none = []
        instances_key_preprocess_normalized = []
        read_items = 0
        accepted_items = 0

        for mxl in mxl_files:
            read_items += 1
            ignored = False
            bwv = None
            if 'Chorales' in self._dataset:
                bwv = mxl[1]
                org_score = m21.corpus.parse(bwv)
                mxl = dir_mxl / Path('Chorales{}_score.mxl'.format(str(mxl[0]).zfill(3)))  # riemen
                org_keysig_sharps, org_keysig_valid = self._check_key_signatures(org_score)
                deb_keysig_sharps, deb_keysig_valid = self._check_key_signatures(m21.converter.parse(mxl))
                if org_keysig_valid and deb_keysig_valid and (org_keysig_sharps == deb_keysig_sharps):
                    pass  # ok
                else:
                    logger.info('Inconsistent key signatures (music21 and WiR Chorales): {}'.format(mxl))
                    org_score = None
                    ignored = True
            else:
                try:
                    org_score = m21.converter.parse(mxl)
                except:
                    logger.info('Read error: {}'.format(mxl))
                    org_score = None
                    ignored = True

            mxl_filename = mxl.stem
            if not ignored:
                rntxt = m21.converter.parse('{}_analysis.rntxt'.format(str(mxl)[:-(len('_score') + len(mxl.suffix))]))
            else:
                rntxt = None

            # Key changes within a piece are not allowed.
            # Not allowed to have different key signatures between parts.
            # Currently, transposing instruments are not supported.
            if not ignored:
                key_sigs_sharps, valid_key_sig = self._check_key_signatures(org_score)
            else:
                key_sigs_sharps, valid_key_sig = None, False

            if ignored or (not self._check_measure_alignment(org_score)) or (not valid_key_sig) or (org_score.parts[0].measure(2) is None):
                logger.info('Ignore {}'.format(mxl_filename))
                continue

            # score => graph
            (
                graph,
                timesig,
                resolution,
                time_per_measure,
                voice_coverage,
                offset2mn,
                raw_ann,
                fermata_positions
            ) = self._get_score_as_graph(org_score)
            fermata_positions = sorted(set(fermata_positions))

            # measure
            timestep_indices = torch.where(graph.x[:, NODE_LABELS['node_type']].long() == TIME_NODE_TYPE_LABEL)[0]
            timesteps = torch.index_select(graph.x, index=timestep_indices, dim=0)[:, NODE_FEATS['offset']].tolist()
            measures = []
            offset2mn_keys = list(offset2mn.keys())
            offset2mn_values = list(offset2mn.values())
            for t in timesteps:
                o_diff = t - np.array(offset2mn_keys)
                o_diff = np.where(
                    o_diff < 0.0,
                    np.ones_like(o_diff) * (o_diff.max() + 1),
                    o_diff
                )
                oi = o_diff.argmin()
                measures.append(offset2mn_values[oi]['m_number'])

            # annotation
            ann = {}
            assert len(rntxt.parts) == 1
            if not self._check_measure_offset_consistency(logger, offset2mn, rntxt):
                # inconsistent measure or offset, skip the score
                logger.info('Ignore {}'.format(mxl_filename))
                continue

            for m in rntxt.parts[0].getElementsByClass(Measure):
                for e in m.elements:
                    if isinstance(e, m21.roman.RomanNumeral):
                        ann[m.offset + e.offset] = {
                            'offset': m.offset + e.offset,
                            'measure': m.measureNumber,
                            'rn': e
                        }

            # nct_ann
            if bool(ann):
                nct_ann = self._get_nct_annotation(graph=graph, ann=ann, logger=logger)
                graph.y = nct_ann
                nct_ratio = (
                        (nct_ann == NCT_NONCHORD_INDEX).long().sum() + (nct_ann == NCT_BOTH_INDEX).long().sum()
                ) / ((nct_ann == NCT_NONCHORD_INDEX).long().sum() + (nct_ann == NCT_BOTH_INDEX).long().sum() + (nct_ann == NCT_CHORD_INDEX).long().sum())
                # debug nct_ann (note color changed)
                org_score = self.generate_nct_score(org_score, graph.x, nct_ann, nct_ann)
                org_score.write('musicxml', Path(self._dir_output_dataset) / Path('nct-score') / Path('{}_nct.musicxml'.format(mxl.stem)))
            else:
                nct_ratio = 0.0
                nct_ann = None

            if self._alert_nct_ratio_th <= nct_ratio:
                logger.info('Ignored {}, NCT-ratio={}, something may be inconsistent'.format(mxl.stem, nct_ratio))
                continue

            accepted_items += 1
            logger.info('Accepted {}'.format(mxl_filename))

            split_no = accepted_items % self._cv_num_set
            split_dict[split_no].append(mxl_filename)
            # instances
            metadata = {
                    'filename': mxl_filename,
                    'full_filename': bwv if 'Chorales' in self._dataset else str(mxl),
                    'key_sigs_sharps': key_sigs_sharps,
                    'timesig_numerator': int(timesig.numerator),
                    'timesig_denominator': int(timesig.denominator),
                    'resolution': resolution,
                    'time_per_measure': time_per_measure,
                    'voice_coverage': voice_coverage,
                    'measure': measures,
                    'ann': ann,
                    'has_nct_ann': nct_ann is not None,
                    'nct_ratio': nct_ratio,
                    'fermata': fermata_positions
                }
            instances_key_preprocess_none.append({
                META_DATA: metadata,
                'graph': graph
            })

            # key preprocess normalized
            transposed_graph = copy.deepcopy(graph)
            interval = Interval(
                m21.key.KeySignature(sharps=key_sigs_sharps).getScale('major').tonic, Pitch('C'))
            for nid in range(transposed_graph.x.size(0)):
                if ((transposed_graph.x[nid, NODE_LABELS['node_type']] == NOTE_NODE_TYPE_LABEL) and
                        (0 <= transposed_graph.x[nid, NODE_FEATS['pc']])):
                    org_midi = Pitch(midi=transposed_graph.x[nid, NODE_FEATS['midi']].item())
                    org_pc = Pitch(pitchClass=transposed_graph.x[nid, NODE_FEATS['pc']].item())
                    transposed_midi = org_midi.transpose(interval).midi
                    transposed_pc = org_pc.transpose(interval).pitchClass
                    assert (transposed_midi % 12) == transposed_pc
                    transposed_graph.x[nid, NODE_FEATS['midi']] = transposed_midi
                    transposed_graph.x[nid, NODE_FEATS['pc']] = transposed_pc

            if bool(ann):
                # nct labels are not changed even if the key signature is changed.
                transposed_graph.y = copy.deepcopy(nct_ann)
            instances_key_preprocess_normalized.append({
                META_DATA: copy.deepcopy(metadata),
                'graph': transposed_graph
            })

        instances = {
            'dataset': self._dataset,
            'cv_num_set': self._cv_num_set,
            'resolution_str': self._resolution_str,
            'chord_type': self._chord_type,
            'alert_nct_ratio_th': self._alert_nct_ratio_th,
            'dir_output_dataset': self._dir_output_dataset,
            KEY_PREPROCESS_NONE: instances_key_preprocess_none,
            KEY_PREPROCESS_NORMALIZE: instances_key_preprocess_normalized,
            'splits': split_dict
        }
        for k, v in split_dict.items():
            logger.info('{}: pieces={}'.format(k, len(v)))
        return instances

    def get_train_dev_test_instances(self, logger, instances, splits, cv_set_no):
        test_instances = []
        dev_instances = []
        train_instances = []
        assert 0 <= cv_set_no < self._cv_num_set
        k_test = cv_set_no
        k_dev = (cv_set_no + 1) % self._cv_num_set
        logger.info('cv{}/{}'.format(cv_set_no, self._cv_num_set))
        for instance in instances:
            if instance[META_DATA]['filename'] in splits[k_test]:
                test_instances.append(instance)
            elif instance[META_DATA]['filename'] in splits[k_dev]:
                dev_instances.append(instance)
            else:
                train_instances.append(instance)
        logger.info('train-instances: {}, dev-instances: {}, test-instances: {}'.format(
            len(train_instances), len(dev_instances), len(test_instances)))
        return train_instances, dev_instances, test_instances
