package org.vadere.state.attributes.models;

import java.util.Arrays;
import java.util.List;

import org.vadere.annotation.factories.attributes.ModelAttributeClass;
import org.vadere.state.attributes.Attributes;

@ModelAttributeClass
public class AttributesSIRG extends Attributes {

	private int infectionsAtStart = 0;
	// The infection rate per 1s. Scaled to the elapsed time.
	private double infectionRate = 0.01;

	// The recovery rate per 1s of infected pedestrians. Scaled to the elapsed time.
	private double recoveryRate = 0.01;
	private double infectionMaxDistance = 1;

	public int getInfectionsAtStart() { return infectionsAtStart; }

	public double getInfectionRate() {
		return infectionRate;
	}

	public double getInfectionMaxDistance() {
		return infectionMaxDistance;
	}

	public double getRecoveryRate() {
		return recoveryRate;
	}

}
