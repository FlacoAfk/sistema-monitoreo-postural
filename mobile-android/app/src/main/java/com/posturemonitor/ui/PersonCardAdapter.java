package com.posturemonitor.ui;

import android.graphics.Color;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import com.google.android.material.card.MaterialCardView;
import androidx.recyclerview.widget.RecyclerView;

import com.posturemonitor.R;
import com.posturemonitor.model.PostureAlert;

import java.util.ArrayList;
import java.util.List;

/**
 * Adapter para mostrar tarjetas de postura por persona.
 */
public class PersonCardAdapter extends RecyclerView.Adapter<PersonCardAdapter.ViewHolder> {

    private final List<PostureAlert> persons = new ArrayList<>();

    public void updatePerson(PostureAlert alert) {
        // Find existing or add new
        int index = -1;
        for (int i = 0; i < persons.size(); i++) {
            if (persons.get(i).personId == alert.personId) {
                index = i;
                break;
            }
        }
        if (index >= 0) {
            persons.set(index, alert);
        } else {
            persons.add(alert);
        }
        notifyDataSetChanged();
    }

    public void removePerson(int personId) {
        for (int i = 0; i < persons.size(); i++) {
            if (persons.get(i).personId == personId) {
                persons.remove(i);
                notifyItemRemoved(i);
                return;
            }
        }
    }

    public void clear() {
        persons.clear();
        notifyDataSetChanged();
    }

    public boolean hasPersons() {
        return !persons.isEmpty();
    }

    @NonNull
    @Override
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_person_card, parent, false);
        return new ViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        PostureAlert alert = persons.get(position);
        holder.bind(alert);
    }

    @Override
    public int getItemCount() {
        return persons.size();
    }

    static class ViewHolder extends RecyclerView.ViewHolder {
        private final TextView personLabel;
        private final TextView statusBadge;
        private final TextView cpiValue;
        private final TextView lumbarValue;
        private final TextView curvatureValue;
        private final TextView badTimeValue;
        private final TextView confidenceValue;
        private final MaterialCardView card;

        ViewHolder(@NonNull View itemView) {
            super(itemView);
            personLabel = itemView.findViewById(R.id.person_label);
            statusBadge = itemView.findViewById(R.id.status_badge);
            cpiValue = itemView.findViewById(R.id.cpi_value);
            lumbarValue = itemView.findViewById(R.id.lumbar_value);
            curvatureValue = itemView.findViewById(R.id.curvature_value);
            badTimeValue = itemView.findViewById(R.id.bad_time_value);
            confidenceValue = itemView.findViewById(R.id.confidence_value);
            card = itemView.findViewById(R.id.person_card);
        }

        void bind(PostureAlert alert) {
            personLabel.setText("Persona " + (alert.personId + 1));
            statusBadge.setText(alert.getBadgeText());
            statusBadge.setTextColor(Color.parseColor(alert.getBadgeColor()));

            cpiValue.setText(String.format("%.1f", alert.cpi));
            lumbarValue.setText(String.format("%.1f°", alert.lumbar));
            curvatureValue.setText(String.format("%.1f%%", alert.curvature));
            badTimeValue.setText(String.format("%.0fs", alert.badTime));
            confidenceValue.setText(String.format("%.0f%%", alert.confidence * 100));

            // Highlight alert cards
            if (alert.isAlert()) {
                card.setCardBackgroundColor(Color.parseColor("#1a0f172a"));
                card.setStrokeWidth(2);
                card.setStrokeColor(Color.parseColor(alert.getBadgeColor()));
            } else {
                card.setCardBackgroundColor(Color.parseColor("#0f172a"));
                card.setStrokeWidth(1);
                card.setStrokeColor(Color.parseColor("#1e293b"));
            }
        }
    }
}
